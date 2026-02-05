use rspirv::dr::{self, Instruction};
use rspirv::spirv;
use std::collections::HashMap;

use super::types::*;
use super::names::extract_id_ref;
use super::HEADER_WORD_COUNT;

pub(super) fn collect_instruction_offsets(words: &[u32]) -> Vec<u32> {
    if words.len() <= HEADER_WORD_COUNT {
        return Vec::new();
    }
    let mut offsets = Vec::new();
    let mut index = HEADER_WORD_COUNT;
    let mut byte_offset = (HEADER_WORD_COUNT * 4) as u32;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        if word_count == 0 || index + word_count > words.len() {
            break;
        }
        offsets.push(byte_offset);
        index += word_count;
        byte_offset += (word_count * 4) as u32;
    }
    offsets
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LocationPlacement {
    BeforeLabel {
        function_index: usize,
        label_index: usize,
    },
    ModuleTail,
}

pub(super) fn classify_location_debugs(words: &[u32], function_count: usize) -> Vec<LocationPlacement> {
    let mut placements = Vec::new();
    if words.len() <= HEADER_WORD_COUNT {
        return placements;
    }
    let mut index = HEADER_WORD_COUNT;
    let mut current_function = 0usize;
    let mut inside_function = false;
    let mut label_index = 0usize;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        if word_count == 0 || index + word_count > words.len() {
            break;
        }
        let opcode = word & 0xFFFF;
        if let Some(op) = spirv::Op::from_u32(opcode) {
            if rspirv::grammar::reflect::is_location_debug(op) {
                if inside_function {
                    placements.push(LocationPlacement::BeforeLabel {
                        function_index: current_function,
                        label_index,
                    });
                } else if current_function >= function_count {
                    placements.push(LocationPlacement::ModuleTail);
                }
            } else {
                match op {
                    spirv::Op::Function => {
                        inside_function = true;
                        label_index = 0;
                    }
                    spirv::Op::Label => {
                        label_index += 1;
                    }
                    spirv::Op::FunctionEnd => {
                        inside_function = false;
                        if current_function < function_count {
                            current_function += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        index += word_count;
    }
    placements
}

pub(super) fn collect_module_instructions<'a>(
    module: &'a dr::Module,
    words: &[u32],
    reorder_blocks: bool,
) -> Vec<InstructionRecord<'a>> {
    let mut records = Vec::new();
    for instruction in &module.capabilities {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.extensions {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.ext_inst_imports {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    if let Some(inst) = module.memory_model.as_ref() {
        records.push(InstructionRecord::new(
            inst,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.entry_points {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.execution_modes {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Header,
        ));
    }
    for instruction in &module.debug_string_source {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.debug_names {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.debug_module_processed {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Debug,
        ));
    }
    for instruction in &module.annotations {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Annotations,
        ));
    }
    let trailing_location_start = module
        .types_global_values
        .iter()
        .rposition(|inst| !rspirv::grammar::reflect::is_location_debug(inst.class.opcode))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let (type_values, trailing_location_insts) =
        module.types_global_values.split_at(trailing_location_start);
    let location_placements = classify_location_debugs(words, module.functions.len());
    let mut placement_iter = location_placements.into_iter();
    let mut label_locations: Vec<Vec<Vec<&Instruction>>> = vec![Vec::new(); module.functions.len()];
    let mut function_tail_locations: Vec<Vec<&Instruction>> =
        vec![Vec::new(); module.functions.len()];
    let mut module_tail_locations: Vec<&Instruction> = Vec::new();

    for instruction in trailing_location_insts {
        match placement_iter
            .next()
            .unwrap_or(LocationPlacement::ModuleTail)
        {
            LocationPlacement::BeforeLabel {
                function_index,
                label_index,
            } => {
                if let Some(buckets) = label_locations.get_mut(function_index) {
                    if buckets.len() <= label_index {
                        buckets.resize_with(label_index + 1, Vec::new);
                    }
                    buckets[label_index].push(instruction);
                } else if let Some(tails) = function_tail_locations.get_mut(function_index) {
                    tails.push(instruction);
                } else {
                    module_tail_locations.push(instruction);
                }
            }
            LocationPlacement::ModuleTail => module_tail_locations.push(instruction),
        }
    }

    for instruction in type_values {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Types,
        ));
    }
    for (function_index, function) in module.functions.iter().enumerate() {
        if let Some(ref def) = function.def {
            records.push(InstructionRecord::new(
                def,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
        for parameter in &function.parameters {
            records.push(InstructionRecord::new(
                parameter,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
        let nest_levels = compute_block_nest_levels(function);
        let block_indices = if reorder_blocks {
            reorder_function_blocks(function)
        } else {
            (0..function.blocks.len()).collect()
        };
        let mut label_index = 0usize;
        for index in block_indices {
            let block = &function.blocks[index];
            let block_depth = nest_levels.get(index).copied().unwrap_or(0);
            if let Some(ref label) = block.label {
                if let Some(buckets) = label_locations.get(function_index) {
                    if let Some(locations) = buckets.get(label_index) {
                        for instruction in locations {
                            records.push(InstructionRecord::new(
                                instruction,
                                block_depth,
                                BlockPosition::Global,
                                ModuleSection::Functions,
                            ));
                        }
                    }
                }
                records.push(InstructionRecord::new(
                    label,
                    block_depth,
                    BlockPosition::Label,
                    ModuleSection::Functions,
                ));
                label_index += 1;
            }
            for instruction in &block.instructions {
                records.push(InstructionRecord::new(
                    instruction,
                    block_depth,
                    BlockPosition::Body,
                    ModuleSection::Functions,
                ));
            }
        }

        if let Some(buckets) = label_locations.get(function_index) {
            if label_index < buckets.len() {
                if let Some(tails) = function_tail_locations.get_mut(function_index) {
                    for locations in &buckets[label_index..] {
                        tails.extend(locations.iter().copied());
                    }
                }
            }
        }

        if let Some(ref end) = function.end {
            if let Some(tails) = function_tail_locations.get(function_index) {
                for instruction in tails {
                    records.push(InstructionRecord::new(
                        instruction,
                        0,
                        BlockPosition::Global,
                        ModuleSection::Functions,
                    ));
                }
            }
            records.push(InstructionRecord::new(
                end,
                0,
                BlockPosition::Global,
                ModuleSection::Functions,
            ));
        }
    }

    for instruction in module_tail_locations {
        records.push(InstructionRecord::new(
            instruction,
            0,
            BlockPosition::Global,
            ModuleSection::Functions,
        ));
    }

    records
}

pub(super) struct CommentAligner {
    pub(super) last_alignment: usize,
}

impl CommentAligner {
    pub(super) fn new() -> Self {
        Self { last_alignment: 0 }
    }

    pub(super) fn append_comment(&mut self, line: &mut String, comment: &str) {
        let line_length = line.chars().count();
        let mut align = line_length + 2;
        align = align.max(self.last_alignment).max(super::COMMENT_COLUMN);
        align = (align + 3) & !0x3;
        self.last_alignment = align.min(super::MAX_COMMENT_ALIGN);
        if line_length < align {
            line.push_str(&" ".repeat(align - line_length));
        }
        line.push_str("; ");
        line.push_str(comment);
    }

    pub(super) fn reset(&mut self) {
        self.last_alignment = 0;
    }
}

pub(super) struct CommentCollector {
    pub(super) enabled: bool,
    pub(super) decorations: HashMap<u32, String>,
}

impl CommentCollector {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            decorations: HashMap::new(),
        }
    }

    pub(super) fn observe(&mut self, instruction: &Instruction) {
        if !self.enabled {
            return;
        }
        if instruction.class.opcode == spirv::Op::Decorate {
            self.record_decorate(instruction);
        }
    }

    pub(super) fn inline_comment(&self, instruction: &Instruction) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if instruction.class.opcode == spirv::Op::Name {
            if let Some(id) = instruction.operands.first().and_then(extract_id_ref) {
                return Some(format!("id %{}", id));
            }
        }
        None
    }

    pub(super) fn result_comment(&self, instruction: &Instruction) -> Option<String> {
        if !self.enabled {
            return None;
        }
        instruction
            .result_id
            .and_then(|id| self.decorations.get(&id).cloned())
    }

    fn record_decorate(&mut self, instruction: &Instruction) {
        let Some(target) = instruction.operands.first().and_then(extract_id_ref) else {
            return;
        };
        let mut parts = Vec::new();
        for operand in instruction.operands.iter().skip(1) {
            parts.push(operand.to_string());
        }
        if parts.is_empty() {
            return;
        }
        let entry = self.decorations.entry(target).or_default();
        if !entry.is_empty() {
            entry.push_str(", ");
        }
        entry.push_str(&parts.join(" "));
    }
}

#[derive(Clone, Copy)]
pub(super) struct StackEntry {
    pub(super) index: usize,
    pub(super) post_visit: bool,
}

pub(super) fn reorder_function_blocks(function: &dr::Function) -> Vec<usize> {
    if function.blocks.is_empty() {
        return Vec::new();
    }

    let (mut infos, id_to_index) = build_block_infos(function);
    let mut stack = Vec::new();
    let mut post_order = Vec::with_capacity(function.blocks.len());
    let mut visited = vec![false; infos.len()];

    if let Some(info) = infos.first_mut() {
        info.nest_level = Some(0);
        info.reachable = true;
    }
    stack.push(StackEntry {
        index: 0,
        post_visit: false,
    });

    while let Some(entry) = stack.pop() {
        if entry.post_visit {
            post_order.push(entry.index);
            continue;
        }

        if visited.get(entry.index).copied().unwrap_or(false) {
            continue;
        }
        if let Some(flag) = visited.get_mut(entry.index) {
            *flag = true;
        }
        stack.push(StackEntry {
            index: entry.index,
            post_visit: true,
        });

        nest_successors(&mut infos, entry.index, &id_to_index);
        infos[entry.index].reachable = true;

        let block = &infos[entry.index];
        // Push higher-priority successors first; reverse post-order traversal will then
        // print structured bodies before their merges.
        push_successor(&mut stack, &id_to_index, block.true_block_id);
        push_successor(&mut stack, &id_to_index, block.false_block_id);
        push_successor(&mut stack, &id_to_index, block.body_block_id);
        push_successor(&mut stack, &id_to_index, block.next_block_id);
        for &case in &block.case_block_ids {
            push_successor(&mut stack, &id_to_index, case);
        }
        push_successor(&mut stack, &id_to_index, block.continue_block_id);
        push_successor(&mut stack, &id_to_index, block.merge_block_id);
    }

    let mut order: Vec<usize> = post_order.into_iter().rev().collect();
    for (index, info) in infos.iter_mut().enumerate() {
        if !info.reachable {
            info.nest_level = Some(0);
            order.push(index);
        }
    }
    order
}

pub(super) fn push_successor(stack: &mut Vec<StackEntry>, id_to_index: &HashMap<u32, usize>, block_id: u32) {
    if block_id == 0 {
        return;
    }
    if let Some(&index) = id_to_index.get(&block_id) {
        stack.push(StackEntry {
            index,
            post_visit: false,
        });
    }
}

pub(super) fn nest_successors(infos: &mut [BlockInfo], index: usize, id_to_index: &HashMap<u32, usize>) {
    let level = infos[index].nest_level.unwrap_or(0);
    let merge_block_id = infos[index].merge_block_id;
    let continue_block_id = infos[index].continue_block_id;
    let true_block_id = infos[index].true_block_id;
    let false_block_id = infos[index].false_block_id;
    let body_block_id = infos[index].body_block_id;
    let next_block_id = infos[index].next_block_id;
    let case_block_ids = infos[index].case_block_ids.clone();

    let mut assign = |target: u32, new_level: u32| {
        if target == 0 {
            return;
        }
        if let Some(&succ_index) = id_to_index.get(&target) {
            if infos[succ_index].nest_level.is_none() {
                infos[succ_index].nest_level = Some(new_level);
            }
        }
    };

    assign(merge_block_id, level);
    assign(continue_block_id, level + 1);
    assign(true_block_id, level + 2);
    assign(false_block_id, level + 2);
    assign(body_block_id, level + 2);
    assign(next_block_id, level);
    for case in &case_block_ids {
        assign(*case, level + 2);
    }
}

#[derive(Default)]
pub(super) struct BlockInfo {
    pub(super) label_id: u32,
    pub(super) merge_block_id: u32,
    pub(super) continue_block_id: u32,
    pub(super) true_block_id: u32,
    pub(super) false_block_id: u32,
    pub(super) body_block_id: u32,
    pub(super) next_block_id: u32,
    pub(super) case_block_ids: Vec<u32>,
    pub(super) nest_level: Option<u32>,
    pub(super) reachable: bool,
}

pub(super) fn compute_block_nest_levels(function: &dr::Function) -> Vec<u32> {
    let block_count = function.blocks.len();
    if block_count == 0 {
        return Vec::new();
    }

    let (mut infos, id_to_index) = build_block_infos(function);
    let mut stack = Vec::new();

    infos[0].nest_level = Some(0);
    stack.push(0usize);

    while let Some(index) = stack.pop() {
        let level = infos[index].nest_level.unwrap_or(0);
        let merge_block_id = infos[index].merge_block_id;
        let continue_block_id = infos[index].continue_block_id;
        let true_block_id = infos[index].true_block_id;
        let false_block_id = infos[index].false_block_id;
        let body_block_id = infos[index].body_block_id;
        let next_block_id = infos[index].next_block_id;
        let case_block_ids = infos[index].case_block_ids.clone();
        let mut assign = |target: u32, new_level: u32| {
            if target == 0 {
                return;
            }
            if let Some(&succ_index) = id_to_index.get(&target) {
                if infos[succ_index].nest_level.is_none() {
                    infos[succ_index].nest_level = Some(new_level);
                    stack.push(succ_index);
                }
            }
        };

        assign(merge_block_id, level);
        assign(continue_block_id, level + 1);
        assign(true_block_id, level + 2);
        assign(false_block_id, level + 2);
        assign(body_block_id, level + 2);
        assign(next_block_id, level);
        for case_id in case_block_ids {
            assign(case_id, level + 2);
        }
    }

    infos
        .into_iter()
        .map(|info| info.nest_level.unwrap_or(0))
        .collect()
}

pub(super) fn build_block_infos(function: &dr::Function) -> (Vec<BlockInfo>, HashMap<u32, usize>) {
    let mut infos = Vec::with_capacity(function.blocks.len());
    let mut id_to_index = HashMap::new();
    for (index, block) in function.blocks.iter().enumerate() {
        let info = build_block_info(block);
        if info.label_id != 0 {
            id_to_index.insert(info.label_id, index);
        }
        infos.push(info);
    }
    (infos, id_to_index)
}

pub(super) fn build_block_info(block: &dr::Block) -> BlockInfo {
    let mut info = BlockInfo {
        label_id: block
            .label
            .as_ref()
            .and_then(|inst| inst.result_id)
            .unwrap_or(0),
        ..BlockInfo::default()
    };
    for instruction in &block.instructions {
        match instruction.class.opcode {
            spirv::Op::LoopMerge => {
                info.merge_block_id = instruction
                    .operands
                    .first()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
                info.continue_block_id = instruction
                    .operands
                    .get(1)
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
            }
            spirv::Op::SelectionMerge => {
                info.merge_block_id = instruction
                    .operands
                    .first()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
            }
            _ => {}
        }
    }

    if let Some(terminator) = block.instructions.last() {
        match terminator.class.opcode {
            spirv::Op::Branch => {
                let target = terminator
                    .operands
                    .last()
                    .and_then(extract_id_ref)
                    .unwrap_or(0);
                if info.merge_block_id != 0 {
                    info.body_block_id = target;
                } else {
                    info.next_block_id = target;
                }
            }
            spirv::Op::BranchConditional => {
                if terminator.operands.len() >= 3 {
                    info.true_block_id = terminator
                        .operands
                        .get(1)
                        .and_then(extract_id_ref)
                        .unwrap_or(0);
                    info.false_block_id = terminator
                        .operands
                        .get(2)
                        .and_then(extract_id_ref)
                        .unwrap_or(0);
                }
            }
            spirv::Op::Switch => {
                for (index, operand) in terminator.operands.iter().enumerate().skip(1) {
                    if index % 2 == 1 {
                        if let Some(id) = extract_id_ref(operand) {
                            info.case_block_ids.push(id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    info
}
