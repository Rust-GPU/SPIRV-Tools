use rspirv::dr::{self, Instruction, Operand};
use rspirv::spirv;
use std::collections::HashMap;

use super::formatting::{
    canonical_storage_class, format_f32_literal, format_f64_literal, format_hex_float,
    format_integer_bits, HEX_FLOAT_F16,
};
use super::types::*;
use super::STANDARD_INDENT_COLUMN;

#[derive(Default)]
pub(super) struct FriendlyNameTable {
    pub(super) names: HashMap<u32, String>,
}

impl FriendlyNameTable {
    pub(super) fn from_module(module: &dr::Module, type_table: &TypeTable) -> Self {
        let mut builder = FriendlyNameBuilder::new(type_table);
        visit_module_instructions(module, |instruction| builder.observe(instruction));
        Self {
            names: builder.finish(),
        }
    }

    pub(super) fn lookup(&self, id: u32) -> Option<&str> {
        self.names.get(&id).map(|name| name.as_str())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

pub(super) fn visit_module_instructions<'a>(
    module: &'a dr::Module,
    mut visit: impl FnMut(&'a Instruction),
) {
    for instruction in &module.capabilities {
        visit(instruction);
    }
    for instruction in &module.extensions {
        visit(instruction);
    }
    for instruction in &module.ext_inst_imports {
        visit(instruction);
    }
    if let Some(inst) = module.memory_model.as_ref() {
        visit(inst);
    }
    for instruction in &module.entry_points {
        visit(instruction);
    }
    for instruction in &module.execution_modes {
        visit(instruction);
    }
    for instruction in &module.debug_string_source {
        visit(instruction);
    }
    for instruction in &module.debug_names {
        visit(instruction);
    }
    for instruction in &module.debug_module_processed {
        visit(instruction);
    }
    for instruction in &module.annotations {
        visit(instruction);
    }
    for instruction in &module.types_global_values {
        visit(instruction);
    }
    for function in &module.functions {
        if let Some(def) = function.def.as_ref() {
            visit(def);
        }
        for parameter in &function.parameters {
            visit(parameter);
        }
        for block in &function.blocks {
            if let Some(label) = block.label.as_ref() {
                visit(label);
            }
            for instruction in &block.instructions {
                visit(instruction);
            }
        }
        if let Some(end) = function.end.as_ref() {
            visit(end);
        }
    }
}

const FP_ENCODING_BFLOAT16_KHR: u32 = 0;
const FP_ENCODING_FLOAT8_E4M3_EXT: u32 = 4214;
const FP_ENCODING_FLOAT8_E5M2_EXT: u32 = 4215;

pub(super) struct FriendlyNameBuilder<'a> {
    pub(super) names: HashMap<u32, String>,
    pub(super) used: HashMap<String, u32>,
    pub(super) type_table: &'a TypeTable,
}

impl<'a> FriendlyNameBuilder<'a> {
    pub(super) fn new(type_table: &'a TypeTable) -> Self {
        Self {
            names: HashMap::new(),
            used: HashMap::new(),
            type_table,
        }
    }

    pub(super) fn observe(&mut self, instruction: &Instruction) {
        use spirv::Op;
        match instruction.class.opcode {
            Op::Name => self.handle_name(instruction),
            Op::Decorate => self.handle_decorate(instruction),
            Op::TypeVoid => self.assign_result_name(instruction, "void"),
            Op::TypeBool => self.assign_result_name(instruction, "bool"),
            Op::TypeInt => self.handle_type_int(instruction),
            Op::TypeFloat => self.handle_type_float(instruction),
            Op::TypeVector => self.handle_type_vector(instruction),
            Op::TypeMatrix => self.handle_type_matrix(instruction),
            Op::TypeArray => self.handle_type_array(instruction, "_arr_"),
            Op::TypeRuntimeArray => self.handle_runtime_array(instruction, "_runtimearr_"),
            Op::TypePointer => self.handle_type_pointer(instruction),
            Op::TypePipe => self.handle_type_pipe(instruction),
            Op::TypeEvent => self.assign_result_name(instruction, "Event"),
            Op::TypeDeviceEvent => self.assign_result_name(instruction, "DeviceEvent"),
            Op::TypeReserveId => self.assign_result_name(instruction, "ReserveId"),
            Op::TypeQueue => self.assign_result_name(instruction, "Queue"),
            Op::TypeOpaque => self.handle_type_opaque(instruction),
            Op::TypePipeStorage => self.assign_result_name(instruction, "PipeStorage"),
            Op::TypeNamedBarrier => self.assign_result_name(instruction, "NamedBarrier"),
            Op::TypeStruct => self.handle_type_struct(instruction),
            Op::ConstantTrue => self.assign_result_name(instruction, "true"),
            Op::ConstantFalse => self.assign_result_name(instruction, "false"),
            Op::Constant => self.handle_constant(instruction),
            _ => {
                if let Some(id) = instruction.result_id {
                    self.ensure_name(id);
                }
            }
        }
    }

    fn handle_name(&mut self, instruction: &Instruction) {
        if let (Some(Operand::IdRef(target)), Some(Operand::LiteralString(raw_name))) =
            (instruction.operands.first(), instruction.operands.get(1))
        {
            self.assign_name(*target, raw_name);
        }
    }

    fn handle_decorate(&mut self, instruction: &Instruction) {
        if instruction.operands.len() < 2 {
            return;
        }
        let Operand::IdRef(target) = instruction.operands[0] else {
            return;
        };
        if let Operand::Decoration(decoration) = instruction.operands[1] {
            if decoration == spirv::Decoration::BuiltIn {
                let built_in_operand = instruction.operands.get(2);
                if let Some(built_in) = built_in_operand.and_then(extract_built_in) {
                    self.assign_builtin_name(target, built_in);
                }
            }
        }
    }

    fn handle_type_int(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(width) = instruction
            .operands
            .first()
            .and_then(literal_operand_to_u32)
        else {
            return;
        };
        let is_signed = instruction
            .operands
            .get(1)
            .and_then(literal_operand_to_u32)
            .map(|value| value != 0)
            .unwrap_or(true);
        match width {
            8 => self.assign_name(result_id, if is_signed { "char" } else { "uchar" }),
            16 => self.assign_name(result_id, if is_signed { "short" } else { "ushort" }),
            32 => self.assign_name(result_id, if is_signed { "int" } else { "uint" }),
            64 => self.assign_name(result_id, if is_signed { "long" } else { "ulong" }),
            _ => {
                let name = if is_signed {
                    format!("i{width}")
                } else {
                    format!("u{width}")
                };
                self.assign_name(result_id, &name);
            }
        }
    }

    fn handle_type_float(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(width) = instruction
            .operands
            .first()
            .and_then(literal_operand_to_u32)
        else {
            return;
        };
        if let Some(encoded) = instruction.operands.get(1).and_then(literal_operand_to_u32) {
            if let Some(name) = fp_encoding_name(encoded) {
                self.assign_name(result_id, name);
                return;
            }
        }
        match width {
            16 => self.assign_name(result_id, "half"),
            32 => self.assign_name(result_id, "float"),
            64 => self.assign_name(result_id, "double"),
            _ => {
                let name = format!("fp{width}");
                self.assign_name(result_id, &name);
            }
        }
    }

    fn handle_type_vector(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(component), Some(count)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(literal_operand_to_u32),
        ) {
            let element_name = self.lookup_name(component);
            let name = format!("v{count}{element_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_matrix(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(column_type), Some(count)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(literal_operand_to_u32),
        ) {
            let column_name = self.lookup_name(column_type);
            let name = format!("mat{count}{column_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_array(&mut self, instruction: &Instruction, prefix: &str) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let (Some(element), Some(length)) = (
            instruction.operands.first().and_then(extract_id_ref),
            instruction.operands.get(1).and_then(extract_id_ref),
        ) {
            let element_name = self.lookup_name(element);
            let length_name = self.lookup_name(length);
            let name = format!("{prefix}{element_name}_{length_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_runtime_array(&mut self, instruction: &Instruction, prefix: &str) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let Some(element) = instruction.operands.first().and_then(extract_id_ref) {
            let element_name = self.lookup_name(element);
            let name = format!("{prefix}{element_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_pointer(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if instruction.operands.len() < 2 {
            return;
        }
        let storage_class = instruction.operands.first().and_then(extract_storage_class);
        let pointee = instruction.operands.get(1).and_then(extract_id_ref);
        if let (Some(class), Some(pointee_id)) = (storage_class, pointee) {
            let pointee_name = self.lookup_name(pointee_id);
            let name = format!("_ptr_{class}_{pointee_name}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_type_pipe(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(access) = instruction
            .operands
            .first()
            .and_then(extract_access_qualifier)
        else {
            return;
        };
        let name = format!("Pipe{access}");
        self.assign_name(result_id, &name);
    }

    fn handle_type_opaque(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        if let Some(name) = instruction
            .operands
            .first()
            .and_then(extract_literal_string)
        {
            let formatted = format!("Opaque_{}", sanitize_identifier(name));
            self.assign_name(result_id, &formatted);
        }
    }

    fn handle_type_struct(&mut self, instruction: &Instruction) {
        if let Some(result_id) = instruction.result_id {
            let name = format!("_struct_{result_id}");
            self.assign_name(result_id, &name);
        }
    }

    fn handle_constant(&mut self, instruction: &Instruction) {
        let Some(result_id) = instruction.result_id else {
            return;
        };
        let Some(type_id) = instruction.result_type else {
            return;
        };
        if let Some(value) = self.constant_literal(instruction) {
            let base = self.lookup_name(type_id);
            let sanitized = value.replace('-', "n");
            let name = format!("{base}_{sanitized}");
            self.assign_name(result_id, &name);
        }
    }

    fn constant_literal(&self, instruction: &Instruction) -> Option<String> {
        let type_id = instruction.result_type?;
        let operand = instruction.operands.first()?;
        let type_info = self.type_table.get(type_id)?;
        match type_info {
            TypeInfo::Int { width, signed } => {
                literal_operand_bits(operand).map(|bits| format_integer_bits(bits, *width, *signed))
            }
            TypeInfo::Float { width } => match width {
                16 => literal_operand_bits(operand)
                    .map(|bits| format_hex_float(bits & 0xffff, &HEX_FLOAT_F16)),
                32 => literal_operand_bits(operand).map(|bits| format_f32_literal(bits as u32)),
                64 => literal_operand_bits(operand).map(format_f64_literal),
                _ => None,
            },
        }
    }

    fn assign_result_name(&mut self, instruction: &Instruction, name: &str) {
        if let Some(result_id) = instruction.result_id {
            self.assign_name(result_id, name);
        }
    }

    fn assign_builtin_name(&mut self, target: u32, built_in: spirv::BuiltIn) {
        if let Some(name) = builtin_name(built_in) {
            self.assign_name(target, name);
        }
    }

    fn assign_name(&mut self, id: u32, raw: &str) {
        if self.names.contains_key(&id) {
            return;
        }
        let base = sanitize_identifier(raw);
        let name = self.unique_name(base);
        self.names.insert(id, name);
    }

    fn ensure_name(&mut self, id: u32) {
        if self.names.contains_key(&id) {
            return;
        }
        let name = id.to_string();
        self.used.entry(name.clone()).or_insert(1);
        self.names.insert(id, name);
    }

    fn lookup_name(&mut self, id: u32) -> String {
        if !self.names.contains_key(&id) {
            self.ensure_name(id);
        }
        self.names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string())
    }

    fn unique_name(&mut self, base: String) -> String {
        let normalized = if base.is_empty() {
            "_".to_owned()
        } else {
            base
        };
        let counter = self.used.entry(normalized.clone()).or_insert(0);
        let result = if *counter == 0 {
            normalized.clone()
        } else {
            format!("{}_{}", normalized, *counter - 1)
        };
        *counter += 1;
        result
    }

    pub(super) fn finish(self) -> HashMap<u32, String> {
        self.names
    }
}

pub(super) fn sanitize_identifier(raw: &str) -> String {
    if raw.is_empty() {
        return "_".to_string();
    }
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn extract_id_ref(operand: &Operand) -> Option<u32> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

pub(super) fn extract_literal_string(operand: &Operand) -> Option<&str> {
    if let Operand::LiteralString(value) = operand {
        Some(value.as_str())
    } else {
        None
    }
}

pub(super) fn extract_built_in(operand: &Operand) -> Option<spirv::BuiltIn> {
    match operand {
        Operand::BuiltIn(value) => Some(*value),
        _ => literal_operand_to_u32(operand).and_then(spirv::BuiltIn::from_u32),
    }
}

pub(super) fn extract_storage_class(operand: &Operand) -> Option<String> {
    match operand {
        Operand::StorageClass(class) => Some(
            canonical_storage_class(*class)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("{:?}", class)),
        ),
        _ => literal_operand_to_u32(operand).map(|value| {
            if let Some(class) = spirv::StorageClass::from_u32(value) {
                canonical_storage_class(class)
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("{:?}", class))
            } else {
                format!("StorageClass{value}")
            }
        }),
    }
}

pub(super) fn extract_access_qualifier(operand: &Operand) -> Option<String> {
    match operand {
        Operand::AccessQualifier(qualifier) => Some(format!("{:?}", qualifier)),
        _ => literal_operand_to_u32(operand).map(|value| {
            spirv::AccessQualifier::from_u32(value)
                .map(|qualifier| format!("{:?}", qualifier))
                .unwrap_or_else(|| format!("AccessQualifier{value}"))
        }),
    }
}

pub(super) fn literal_operand_bits(operand: &Operand) -> Option<u64> {
    match operand {
        Operand::LiteralBit32(value) => Some(u64::from(*value)),
        Operand::LiteralBit64(value) => Some(*value),
        _ => None,
    }
}

pub(super) fn section_heading(section: ModuleSection) -> Option<&'static str> {
    match section {
        ModuleSection::Debug => Some("Debug Information"),
        ModuleSection::Annotations => Some("Annotations"),
        ModuleSection::Types => Some("Types, variables and constants"),
        _ => None,
    }
}

pub(super) fn append_section_heading(text: &mut String, heading: &str, indent: bool) {
    if !text.is_empty() {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push('\n');
    }
    if indent {
        text.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    }
    text.push_str("; ");
    text.push_str(heading);
    text.push('\n');
}

pub(super) fn append_function_heading(
    text: &mut String,
    result_id: Option<u32>,
    indent: bool,
    extra_spacing: bool,
) {
    let Some(id) = result_id else {
        return;
    };
    if !text.is_empty() {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push('\n');
        if extra_spacing {
            text.push('\n');
        }
    }
    if indent {
        text.push_str(&" ".repeat(STANDARD_INDENT_COLUMN));
    }
    text.push_str("; Function ");
    text.push_str(&id.to_string());
    text.push('\n');
}

pub(super) fn fp_encoding_name(value: u32) -> Option<&'static str> {
    match value {
        FP_ENCODING_BFLOAT16_KHR => Some("bfloat16"),
        FP_ENCODING_FLOAT8_E4M3_EXT => Some("fp8e4m3"),
        FP_ENCODING_FLOAT8_E5M2_EXT => Some("fp8e5m2"),
        _ => None,
    }
}

pub(super) fn builtin_name(built_in: spirv::BuiltIn) -> Option<&'static str> {
    match built_in {
        spirv::BuiltIn::Position => Some("gl_Position"),
        spirv::BuiltIn::PointSize => Some("gl_PointSize"),
        spirv::BuiltIn::ClipDistance => Some("gl_ClipDistance"),
        spirv::BuiltIn::CullDistance => Some("gl_CullDistance"),
        spirv::BuiltIn::VertexId => Some("gl_VertexID"),
        spirv::BuiltIn::InstanceId => Some("gl_InstanceID"),
        spirv::BuiltIn::PrimitiveId => Some("gl_PrimitiveID"),
        spirv::BuiltIn::InvocationId => Some("gl_InvocationID"),
        spirv::BuiltIn::Layer => Some("gl_Layer"),
        spirv::BuiltIn::ViewportIndex => Some("gl_ViewportIndex"),
        spirv::BuiltIn::TessLevelOuter => Some("gl_TessLevelOuter"),
        spirv::BuiltIn::TessLevelInner => Some("gl_TessLevelInner"),
        spirv::BuiltIn::TessCoord => Some("gl_TessCoord"),
        spirv::BuiltIn::PatchVertices => Some("gl_PatchVertices"),
        spirv::BuiltIn::FragCoord => Some("gl_FragCoord"),
        spirv::BuiltIn::PointCoord => Some("gl_PointCoord"),
        spirv::BuiltIn::FrontFacing => Some("gl_FrontFacing"),
        spirv::BuiltIn::SampleId => Some("gl_SampleID"),
        spirv::BuiltIn::SamplePosition => Some("gl_SamplePosition"),
        spirv::BuiltIn::SampleMask => Some("gl_SampleMask"),
        spirv::BuiltIn::FragDepth => Some("gl_FragDepth"),
        spirv::BuiltIn::HelperInvocation => Some("gl_HelperInvocation"),
        spirv::BuiltIn::NumWorkgroups => Some("gl_NumWorkGroups"),
        spirv::BuiltIn::WorkgroupSize => Some("gl_WorkGroupSize"),
        spirv::BuiltIn::WorkgroupId => Some("gl_WorkGroupID"),
        spirv::BuiltIn::LocalInvocationId => Some("gl_LocalInvocationID"),
        spirv::BuiltIn::GlobalInvocationId => Some("gl_GlobalInvocationID"),
        spirv::BuiltIn::LocalInvocationIndex => Some("gl_LocalInvocationIndex"),
        spirv::BuiltIn::VertexIndex => Some("gl_VertexIndex"),
        spirv::BuiltIn::InstanceIndex => Some("gl_InstanceIndex"),
        spirv::BuiltIn::BaseVertex => Some("gl_BaseVertex"),
        spirv::BuiltIn::BaseInstance => Some("gl_BaseInstance"),
        spirv::BuiltIn::WorkDim => Some("WorkDim"),
        spirv::BuiltIn::GlobalSize => Some("GlobalSize"),
        spirv::BuiltIn::EnqueuedWorkgroupSize => Some("EnqueuedWorkgroupSize"),
        spirv::BuiltIn::GlobalOffset => Some("GlobalOffset"),
        spirv::BuiltIn::GlobalLinearId => Some("GlobalLinearId"),
        spirv::BuiltIn::SubgroupSize => Some("SubgroupSize"),
        spirv::BuiltIn::SubgroupMaxSize => Some("SubgroupMaxSize"),
        spirv::BuiltIn::NumSubgroups => Some("NumSubgroups"),
        spirv::BuiltIn::NumEnqueuedSubgroups => Some("NumEnqueuedSubgroups"),
        spirv::BuiltIn::SubgroupId => Some("SubgroupId"),
        spirv::BuiltIn::SubgroupLocalInvocationId => Some("SubgroupLocalInvocationId"),
        spirv::BuiltIn::SubgroupEqMask => Some("SubgroupEqMaskKHR"),
        spirv::BuiltIn::SubgroupGeMask => Some("SubgroupGeMaskKHR"),
        spirv::BuiltIn::SubgroupGtMask => Some("SubgroupGtMaskKHR"),
        spirv::BuiltIn::SubgroupLeMask => Some("SubgroupLeMaskKHR"),
        spirv::BuiltIn::SubgroupLtMask => Some("SubgroupLtMaskKHR"),
        _ => None,
    }
}
