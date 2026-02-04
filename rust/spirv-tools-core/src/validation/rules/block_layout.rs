//! Block layout validation rules.
//!
//! This module validates SPIR-V block layout requirements including:
//!
//! - Member offset alignment
//! - Array/matrix stride alignment
//! - Runtime array positioning
//! - Relaxed and scalar block layout support

use std::collections::{HashMap, HashSet};

use rspirv::dr::Module;
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::span::SpannedValidationError;
use crate::validation::types::{MemberIndex, ResultId, TypeId};
use crate::validation::ValidationResult;

// ============================================================================
// Block Layout Rule
// ============================================================================

/// Validates block layout requirements for uniform/storage buffer structs.
pub struct BlockLayoutRule;

impl ValidationRule for BlockLayoutRule {
    fn name(&self) -> &'static str {
        "block-layout"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        ctx.options.skip_block_layout
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let scalar_layout =
            ctx.options.scalar_block_layout || ctx.options.workgroup_scalar_block_layout;
        let relax_block_layout = ctx.options.relax_block_layout;

        let block_structs = collect_block_structs(ctx.module);
        for (struct_id, block_info) in block_structs {
            // Only validate block layout for structs used in storage classes that require it.
            if !block_info.requires_block_layout() {
                continue;
            }

            // Std140 extended alignment (16-byte rounding for arrays/structs) only
            // applies to Uniform + Block decoration. All other combinations
            // (StorageBuffer + Block, BufferBlock + Uniform, PushConstant + Block, etc.)
            // use std430 rules which have no such rounding. This matches the C++
            // SPIRV-Tools behavior in validate_decorations.cpp (blockRules vs bufferRules).
            let is_uniform_block = block_info.decoration == BlockDecoration::Block
                && block_info.storage_classes.contains(&StorageClass::Uniform);
            // In C++, uniform_buffer_standard_layout makes blockRules=false,
            // disabling extended alignment. relax_block_layout does NOT disable it.
            let extended_alignment =
                is_uniform_block && !ctx.options.uniform_buffer_standard_layout;

            let Some(struct_inst) = ctx.definitions.get(&struct_id) else {
                continue;
            };
            if struct_inst.class.opcode != Op::TypeStruct {
                continue;
            }
            if struct_inst.operands.is_empty() {
                continue;
            }
            let member_offsets = collect_member_offsets(ctx.module, struct_id);
            let member_count = struct_inst.operands.len();
            for (index, operand) in struct_inst.operands.iter().enumerate() {
                let Some(offset) = member_offsets.get(&MemberIndex(index as u32)) else {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "missing OpMemberDecorate Offset".to_string(),
                    }
                    .into());
                };
                let rspirv::dr::Operand::IdRef(member_type_id_raw) = operand else {
                    continue;
                };
                let Ok(member_type_id) = TypeId::try_from(*member_type_id_raw) else {
                    continue;
                };
                let Ok(member_result_id) = ResultId::try_from(u32::from(member_type_id)) else {
                    continue;
                };
                let Some(member_inst) = ctx.definitions.get(&member_result_id) else {
                    continue;
                };
                // In C++, scalar_block_layout uses getScalarAlignment, while
                // everything else (including relaxed layout) uses getBaseAlignment
                // with standard vector 2N/4N rules. Only scalar layout changes
                // vector alignment to scalar.
                let Some(alignment) = type_alignment(
                    member_type_id,
                    ctx.definitions,
                    &mut HashSet::new(),
                    scalar_layout,
                    extended_alignment,
                ) else {
                    continue;
                };
                if member_inst.class.opcode == Op::TypeRuntimeArray {
                    if index + 1 != member_count {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: "runtime array member must be the final struct member"
                                .to_string(),
                        }
                        .into());
                    }
                    if let Some(stride) = array_stride(ctx.module, member_result_id) {
                        if stride % alignment != 0 {
                            return Err(ValidationError::InvalidBlockLayout {
                                struct_type: struct_id,
                                reason: format!(
                                    "runtime array stride {stride} is not aligned to {alignment}"
                                ),
                            }
                            .into());
                        }
                    }
                    // Runtime array must be last; remaining checks do not apply.
                    continue;
                }
                if member_inst.class.opcode == Op::TypeArray {
                    if let Some(stride) = array_stride(ctx.module, member_result_id) {
                        if stride % alignment != 0 {
                            return Err(ValidationError::InvalidBlockLayout {
                                struct_type: struct_id,
                                reason: format!(
                                    "array stride {stride} is not aligned to {alignment}"
                                ),
                            }
                            .into());
                        }
                        if let Some(rspirv::dr::Operand::IdRef(elem_raw)) =
                            member_inst.operands.first()
                        {
                            if let Ok(elem_type) = TypeId::try_from(*elem_raw) {
                                if let Some(elem_size) = type_layout_size(
                                    elem_type,
                                    ctx.definitions,
                                    &mut HashSet::new(),
                                ) {
                                    if elem_size > stride {
                                        return Err(ValidationError::InvalidBlockLayout {
                                            struct_type: struct_id,
                                            reason: format!(
                                                "array stride {stride} is smaller than element size {elem_size}"
                                            ),
                                        }.into());
                                    }
                                }
                            }
                        }
                    }
                }
                if member_inst.class.opcode == Op::TypeMatrix {
                    let stride =
                        member_matrix_stride(ctx.module, struct_id, MemberIndex(index as u32))
                            .ok_or_else(|| -> SpannedValidationError {
                                ValidationError::InvalidBlockLayout {
                                    struct_type: struct_id,
                                    reason: "matrix member is missing MatrixStride".to_string(),
                                }
                                .into()
                            })?;
                    if stride % alignment != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!("matrix stride {stride} is not aligned to {alignment}"),
                        }
                        .into());
                    }
                    let (column_type, _) = matrix_info(member_inst);
                    if let Some(col_ty) = column_type {
                        if let Some(col_size) =
                            type_layout_size(col_ty, ctx.definitions, &mut HashSet::new())
                        {
                            if col_size > stride {
                                return Err(ValidationError::InvalidBlockLayout {
                                    struct_type: struct_id,
                                    reason: format!(
                                        "matrix stride {stride} is smaller than column size {col_size}"
                                    ),
                                }.into());
                            }
                            if relax_block_layout
                                && !scalar_layout
                                && member_is_row_major(
                                    ctx.module,
                                    struct_id,
                                    MemberIndex(index as u32),
                                )
                                && col_size > 16
                                && (offset % 16).saturating_add(col_size) > 16
                            {
                                return Err(ValidationError::InvalidBlockLayout {
                                    struct_type: struct_id,
                                    reason: "row-major matrix straddles 16-byte boundary under relaxed layout".to_string(),
                                }.into());
                            }
                        }
                    }
                }
                let Some(size) =
                    type_layout_size(member_type_id, ctx.definitions, &mut HashSet::new())
                else {
                    continue;
                };
                // Offset alignment rules.
                // In C++, relaxed layout only changes vector offset checks to use
                // scalar element alignment. All other types (matrices, arrays,
                // structs, scalars) still use standard alignment for offset checks.
                if relax_block_layout
                    && !scalar_layout
                    && member_inst.class.opcode == Op::TypeVector
                {
                    let Some(scalar_align) = vector_scalar_alignment(member_inst, ctx.definitions)
                    else {
                        continue;
                    };
                    if offset % scalar_align != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "member offset {offset} is not aligned to vector scalar element size {}",
                                scalar_align
                            ),
                        }.into());
                    }
                    let Some(vector_size) =
                        type_layout_size(member_type_id, ctx.definitions, &mut HashSet::new())
                    else {
                        continue;
                    };
                    // From C++ hasImproperStraddle():
                    // - size <= 16: straddles if first and last byte are in
                    //   different 16-byte blocks: (F / 16) != (L / 16)
                    // - size > 16: straddles if not 16-byte aligned: F % 16 != 0
                    let straddles = if vector_size <= 16 {
                        vector_size > 0
                            && (offset >> 4) != ((offset.saturating_add(vector_size - 1)) >> 4)
                    } else {
                        offset % 16 != 0
                    };
                    if straddles {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "vector at offset {offset} improperly straddles a 16-byte boundary"
                            ),
                        }
                        .into());
                    }
                } else if offset % alignment != 0 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: format!(
                            "member offset {offset} is not aligned to required alignment {alignment}"
                        ),
                    }.into());
                }

                let mut next_valid_offset = offset.saturating_add(size);
                // From C++ validate_decorations.cpp: non-scalar block layout
                // rules don't permit anything in the padding of a struct or
                // array. Round up the next valid offset to the member's
                // alignment so that the next member cannot overlap padding.
                if !scalar_layout
                    && matches!(
                        member_inst.class.opcode,
                        Op::TypeArray | Op::TypeStruct
                    )
                {
                    next_valid_offset = round_up(next_valid_offset, alignment);
                }
                // Ensure no overlap with the next member offset (if any).
                if let Some(next) = member_offsets
                    .get(&MemberIndex((index as u32) + 1))
                    .copied()
                {
                    if next < next_valid_offset {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: "member offsets overlap".to_string(),
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Block Info Types
// ============================================================================

/// Which block decoration a struct has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockDecoration {
    Block,
    BufferBlock,
}

/// Information about a struct that may require block layout validation.
#[derive(Debug, Clone)]
struct BlockStructInfo {
    decoration: BlockDecoration,
    storage_classes: HashSet<StorageClass>,
}

impl BlockStructInfo {
    fn new(decoration: BlockDecoration) -> Self {
        Self {
            decoration,
            storage_classes: HashSet::new(),
        }
    }

    /// Returns true if block layout rules apply to this struct based on its
    /// decoration and the storage classes where it's used.
    ///
    /// From the C++ spirv-val (validate_decorations.cpp lines 1365-1369):
    /// - Block rules: Uniform storage class + Block decoration
    /// - Buffer rules: (Uniform + BufferBlock) OR
    ///                 ((PushConstant | StorageBuffer | PhysicalStorageBuffer | Workgroup) + Block)
    ///
    /// Block-decorated structs that are not used in any variable (or used only in
    /// non-buffer storage classes like Output/Input) do NOT require Offset decorations.
    fn requires_block_layout(&self) -> bool {
        let has_uniform = self.storage_classes.contains(&StorageClass::Uniform);
        let has_push_constant = self.storage_classes.contains(&StorageClass::PushConstant);
        let has_storage_buffer = self.storage_classes.contains(&StorageClass::StorageBuffer);
        let has_phys_storage_buffer = self
            .storage_classes
            .contains(&StorageClass::PhysicalStorageBuffer);
        let has_workgroup = self.storage_classes.contains(&StorageClass::Workgroup);

        match self.decoration {
            BlockDecoration::Block => {
                has_uniform
                    || has_push_constant
                    || has_storage_buffer
                    || has_phys_storage_buffer
                    || has_workgroup
            }
            BlockDecoration::BufferBlock => has_uniform,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn collect_block_structs(module: &Module) -> HashMap<ResultId, BlockStructInfo> {
    let mut structs: HashMap<ResultId, BlockStructInfo> = HashMap::new();

    // First pass: collect all Block/BufferBlock decorated structs
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                let block_deco = match *decoration {
                    Decoration::Block => Some(BlockDecoration::Block),
                    Decoration::BufferBlock => Some(BlockDecoration::BufferBlock),
                    _ => None,
                };
                if let Some(deco) = block_deco {
                    if let Ok(struct_id) = ResultId::try_from(*target) {
                        structs
                            .entry(struct_id)
                            .or_insert_with(|| BlockStructInfo::new(deco));
                    }
                }
            }
        }
    }

    // Second pass: map struct ids to storage classes where they are used
    for var in &module.types_global_values {
        if var.class.opcode != Op::Variable {
            continue;
        }
        let Some(rspirv::dr::Operand::StorageClass(sc)) = var.operands.first() else {
            continue;
        };
        let Some(result_type) = var.result_type else {
            continue;
        };
        let Ok(ptr_type_id) = TypeId::try_from(result_type) else {
            continue;
        };
        let Some(ptr_inst) = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(u32::from(ptr_type_id)))
        else {
            continue;
        };
        if ptr_inst.class.opcode != Op::TypePointer {
            continue;
        }
        let Some(rspirv::dr::Operand::IdRef(pointee)) = ptr_inst.operands.get(1) else {
            continue;
        };
        if let Ok(struct_id) = ResultId::try_from(*pointee) {
            if let Some(info) = structs.get_mut(&struct_id) {
                info.storage_classes.insert(*sc);
            }
        }
    }

    structs
}

fn collect_member_offsets(module: &Module, struct_id: ResultId) -> HashMap<MemberIndex, u32> {
    let mut offsets = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            let mut operands = inst.operands.iter();
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::LiteralBit32(member)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (operands.next(), operands.next(), operands.next())
            {
                if *decoration == Decoration::Offset {
                    if let Ok(target_id) = ResultId::try_from(*target) {
                        if target_id == struct_id {
                            if let Some(rspirv::dr::Operand::LiteralBit32(offset)) = operands.next()
                            {
                                offsets.insert(MemberIndex(*member), *offset);
                            }
                        }
                    }
                }
            }
        }
    }
    offsets
}

fn type_layout_size(
    ty: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
) -> Option<u32> {
    if !visiting.insert(ty) {
        return None;
    }
    let inst = definitions.get(&ResultId::try_from(u32::from(ty)).ok()?)?;
    let size = match inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => inst.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::LiteralBit32(bits) => Some(*bits / 8),
            _ => None,
        }),
        Op::TypeVector => {
            let (elem, count) = vector_info(inst);
            let (elem, count) = (elem?, count?);
            let elem_size = type_layout_size(elem, definitions, visiting)?;
            Some(elem_size.saturating_mul(count))
        }
        Op::TypeMatrix => {
            let (column, count) = matrix_info(inst);
            let (column, count) = (column?, count?);
            let col_size = type_layout_size(column, definitions, visiting)?;
            Some(col_size.saturating_mul(count))
        }
        Op::TypeArray => {
            let elem = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            })?;
            let elem_size = type_layout_size(elem, definitions, visiting)?;
            let len = array_length(inst, definitions)?;
            Some(elem_size.saturating_mul(len))
        }
        Op::TypeRuntimeArray => None, // unsized
        Op::TypeStruct => {
            let mut offset: u32 = 0;
            for op in &inst.operands {
                let ty = match op {
                    rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok()?,
                    _ => return None,
                };
                let size = type_layout_size(ty, definitions, visiting)?;
                offset = offset.saturating_add(size);
            }
            Some(offset)
        }
        _ => None,
    };
    visiting.remove(&ty);
    size
}

/// Rounds a value up to the next multiple of align.
#[inline]
fn round_up(value: u32, align: u32) -> u32 {
    if align == 0 {
        value
    } else {
        value.wrapping_add(align.wrapping_sub(1)) / align * align
    }
}

/// Calculates type alignment, matching C++ getBaseAlignment().
///
/// When `extended_alignment` is true (std140, i.e. Uniform + Block), arrays,
/// structs, and matrices have their base alignment rounded up to 16 bytes.
/// When false (std430 or scalar), no rounding is applied.
///
/// When `scalar_layout` is true, vectors use scalar element alignment instead
/// of the standard 2N/4N rules.
fn type_alignment(
    ty: TypeId,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
    scalar_layout: bool,
    extended_alignment: bool,
) -> Option<u32> {
    if !visiting.insert(ty) {
        return None;
    }
    let inst = definitions.get(&ResultId::try_from(u32::from(ty)).ok()?)?;
    let alignment = match inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => inst.operands.first().and_then(|op| match op {
            rspirv::dr::Operand::LiteralBit32(bits) => Some(*bits / 8),
            _ => None,
        }),
        Op::TypeVector => {
            let (elem, count) = vector_info(inst);
            let (elem, count) = (elem?, count?);
            let elem_align =
                type_alignment(elem, definitions, visiting, scalar_layout, extended_alignment)?;
            if scalar_layout {
                Some(elem_align)
            } else {
                // vec2 aligns to 2N, vec3/vec4 align to 4N (where N is scalar alignment)
                let multiplier = if count == 2 { 2 } else { 4 };
                elem_align.checked_mul(multiplier)
            }
        }
        Op::TypeMatrix => {
            // Matrix alignment follows its column vector alignment.
            let (column, _) = matrix_info(inst);
            let column = column?;
            let base_align =
                type_alignment(column, definitions, visiting, scalar_layout, extended_alignment)?;
            if extended_alignment && !scalar_layout {
                Some(round_up(base_align, 16))
            } else {
                Some(base_align)
            }
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            let elem = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            })?;
            let base_align =
                type_alignment(elem, definitions, visiting, scalar_layout, extended_alignment)?;
            if extended_alignment && !scalar_layout {
                Some(round_up(base_align, 16))
            } else {
                Some(base_align)
            }
        }
        Op::TypeStruct => {
            let mut max_align = 1;
            for op in &inst.operands {
                let ty = match op {
                    rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok()?,
                    _ => return None,
                };
                let align =
                    type_alignment(ty, definitions, visiting, scalar_layout, extended_alignment)?;
                max_align = max_align.max(align);
            }
            if extended_alignment && !scalar_layout {
                Some(round_up(max_align, 16))
            } else {
                Some(max_align)
            }
        }
        _ => None,
    };
    visiting.remove(&ty);
    alignment
}

fn vector_scalar_alignment(
    vector_inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let elem = vector_inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    })?;
    type_alignment(elem, definitions, &mut HashSet::new(), true, false)
}

fn vector_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let elem = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(c) => Some(*c),
        _ => None,
    });
    (elem, count)
}

fn matrix_info(inst: &rspirv::dr::Instruction) -> (Option<TypeId>, Option<u32>) {
    let column = inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    });
    let count = inst.operands.get(1).and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(c) => Some(*c),
        _ => None,
    });
    (column, count)
}

fn array_length(
    inst: &rspirv::dr::Instruction,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let len_id = match inst.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => ResultId::try_from(*id).ok()?,
        _ => return None,
    };
    let len_inst = definitions.get(&len_id)?;
    if len_inst.class.opcode != Op::Constant {
        return None;
    }
    match len_inst.operands.first() {
        Some(rspirv::dr::Operand::LiteralBit32(v)) => Some(*v),
        Some(rspirv::dr::Operand::LiteralBit64(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

/// Get the ArrayStride decoration for an array type.
pub fn array_stride(module: &Module, array_type: ResultId) -> Option<u32> {
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
                Some(rspirv::dr::Operand::LiteralBit32(stride)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if *decoration == Decoration::ArrayStride {
                    if let Ok(target_id) = ResultId::try_from(*target) {
                        if target_id == array_type {
                            return Some(*stride);
                        }
                    }
                }
            }
        }
    }
    None
}

fn member_is_row_major(module: &Module, struct_id: ResultId, member: MemberIndex) -> bool {
    for inst in &module.annotations {
        if inst.class.opcode != Op::MemberDecorate {
            continue;
        }
        let mut ops = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = ops.next() else {
            continue;
        };
        let Ok(target_id) = ResultId::try_from(*target) else {
            continue;
        };
        if target_id != struct_id {
            continue;
        }
        let Some(rspirv::dr::Operand::LiteralBit32(member_idx)) = ops.next() else {
            continue;
        };
        if *member_idx != member.0 {
            continue;
        }
        let Some(rspirv::dr::Operand::Decoration(decoration)) = ops.next() else {
            continue;
        };
        if *decoration == Decoration::RowMajor {
            return true;
        }
    }
    false
}

fn member_matrix_stride(module: &Module, struct_id: ResultId, member: MemberIndex) -> Option<u32> {
    for inst in &module.annotations {
        if inst.class.opcode != Op::MemberDecorate {
            continue;
        }
        let mut ops = inst.operands.iter();
        let Some(rspirv::dr::Operand::IdRef(target)) = ops.next() else {
            continue;
        };
        let Ok(target_id) = ResultId::try_from(*target) else {
            continue;
        };
        if target_id != struct_id {
            continue;
        }
        let Some(rspirv::dr::Operand::LiteralBit32(member_idx)) = ops.next() else {
            continue;
        };
        if *member_idx != member.0 {
            continue;
        }
        let Some(rspirv::dr::Operand::Decoration(decoration)) = ops.next() else {
            continue;
        };
        if *decoration != Decoration::MatrixStride {
            continue;
        }
        if let Some(rspirv::dr::Operand::LiteralBit32(stride)) = ops.next() {
            return Some(*stride);
        }
    }
    None
}

// ============================================================================
// All block layout rules
// ============================================================================

/// Returns all block layout validation rules.
pub fn all_block_layout_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&BlockLayoutRule]
}
