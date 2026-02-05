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
use crate::validation::helpers::is_vulkan_env;
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
        let relax_block_layout = ctx.options.relax_block_layout;

        let block_structs = collect_block_structs(ctx.module);
        for (struct_id, block_info) in block_structs {
            if !block_info.requires_block_layout() {
                continue;
            }

            // C++ uses workgroup_scalar_block_layout only for Workgroup storage
            // class, and scalar_block_layout for all others (lines 1428-1430).
            let has_workgroup = block_info
                .storage_classes
                .contains(&StorageClass::Workgroup);
            let scalar_layout = if has_workgroup {
                ctx.options.workgroup_scalar_block_layout
            } else {
                ctx.options.scalar_block_layout
            };

            // Std140 extended alignment (16-byte rounding for arrays/structs) only
            // applies to Uniform + Block decoration. All other combinations use
            // std430 rules. In C++, uniform_buffer_standard_layout makes
            // blockRules=false. relax_block_layout does NOT disable it.
            let is_uniform_block = block_info.decoration == BlockDecoration::Block
                && block_info.storage_classes.contains(&StorageClass::Uniform);
            let extended_alignment =
                is_uniform_block && !ctx.options.uniform_buffer_standard_layout;

            let deco_name = match block_info.decoration {
                BlockDecoration::Block => "Block",
                BlockDecoration::BufferBlock => "BufferBlock",
            };

            // Pre-checks run for ALL environments (C++ lines 1432-1476).
            check_required_block_decorations(
                ctx.module,
                ctx.definitions,
                struct_id,
                deco_name,
                &mut HashSet::new(),
            )?;

            // Actual layout validation (offset alignment, overlap, straddle) only
            // runs for Vulkan environments (C++ line 1478).
            if is_vulkan_env(ctx.env) {
                check_struct_layout(
                    ctx.module,
                    ctx.definitions,
                    struct_id,
                    0,
                    scalar_layout,
                    extended_alignment,
                    relax_block_layout,
                    0,
                )?;
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
        // Jump through one level of array indirection for non-Workgroup
        // storage classes (C++ CheckDecorationsOfBuffers, lines 1354-1360).
        // Variables like ptr-to-array-of-BlockStruct are descriptor arrays;
        // the layout check applies to the struct, not the outer array.
        let mut target = *pointee;
        if *sc != StorageClass::Workgroup {
            if let Ok(tid) = ResultId::try_from(target) {
                if let Some(inner) = module
                    .types_global_values
                    .iter()
                    .find(|i| i.result_id == Some(u32::from(tid)))
                {
                    if inner.class.opcode == Op::TypeArray
                        || inner.class.opcode == Op::TypeRuntimeArray
                    {
                        if let Some(rspirv::dr::Operand::IdRef(elem)) = inner.operands.first() {
                            target = *elem;
                        }
                    }
                }
            }
        }
        if let Ok(struct_id) = ResultId::try_from(target) {
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
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
    is_row_major: bool,
    matrix_stride: u32,
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
            let elem_size = type_layout_size(elem, module, definitions, visiting, false, 0)?;
            Some(elem_size.saturating_mul(count))
        }
        Op::TypeMatrix => {
            let (column, num_columns) = matrix_info(inst);
            let (column, num_columns) = (column?, num_columns?);
            if matrix_stride > 0 {
                if is_row_major {
                    // Row major: (num_rows - 1) * stride + num_columns * scalar_size
                    // (C++ getSize, validate_decorations.cpp lines 364-374)
                    let col_inst =
                        definitions.get(&ResultId::try_from(u32::from(column)).ok()?)?;
                    let (scalar_type, num_rows) = vector_info(col_inst);
                    let (scalar_type, num_rows) = (scalar_type?, num_rows?);
                    let scalar_size =
                        type_layout_size(scalar_type, module, definitions, visiting, false, 0)?;
                    Some(
                        num_rows
                            .saturating_sub(1)
                            .saturating_mul(matrix_stride)
                            .saturating_add(num_columns.saturating_mul(scalar_size)),
                    )
                } else {
                    // Column major: num_columns * stride
                    // (C++ getSize, validate_decorations.cpp lines 362-363)
                    Some(num_columns.saturating_mul(matrix_stride))
                }
            } else {
                // No stride info, fall back to raw computation.
                let col_size =
                    type_layout_size(column, module, definitions, visiting, false, 0)?;
                Some(col_size.saturating_mul(num_columns))
            }
        }
        Op::TypeArray => {
            let elem = inst.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            })?;
            let elem_size =
                type_layout_size(elem, module, definitions, visiting, is_row_major, matrix_stride)?;
            let len = array_length(inst, definitions)?;
            // Use (N-1)*stride + elem_size to account for padding between elements
            // (C++ getSize, validate_decorations.cpp lines 344-357).
            let array_type_id = ResultId::try_from(u32::from(ty)).ok()?;
            let stride = array_stride(module, array_type_id).unwrap_or(0);
            if stride > 0 && len > 0 {
                Some(
                    len.saturating_sub(1)
                        .saturating_mul(stride)
                        .saturating_add(elem_size),
                )
            } else {
                Some(elem_size.saturating_mul(len))
            }
        }
        Op::TypeRuntimeArray => None, // unsized
        Op::TypeStruct => {
            // Struct size = offset_of_last_member + size_of_last_member
            // (C++ getSize, validate_decorations.cpp lines 376-397).
            // This accounts for padding gaps between members specified by
            // Offset decorations, rather than naively summing member sizes.
            let struct_id = ResultId::try_from(u32::from(ty)).ok()?;
            let members: Vec<_> = inst
                .operands
                .iter()
                .enumerate()
                .filter_map(|(idx, op)| match op {
                    rspirv::dr::Operand::IdRef(id) => {
                        TypeId::try_from(*id).ok().map(|ty| (idx as u32, ty))
                    }
                    _ => None,
                })
                .collect();
            if members.is_empty() {
                return Some(0);
            }
            let offsets = collect_member_offsets(module, struct_id);
            // Find the last member by offset order.
            let (last_idx, last_ty) = members
                .iter()
                .max_by_key(|(idx, _)| offsets.get(&MemberIndex(*idx)).copied().unwrap_or(0))?;
            let last_offset = offsets.get(&MemberIndex(*last_idx)).copied()?;
            let last_rm = member_is_row_major(module, struct_id, MemberIndex(*last_idx));
            let last_ms =
                member_matrix_stride(module, struct_id, MemberIndex(*last_idx)).unwrap_or(0);
            let last_size =
                type_layout_size(*last_ty, module, definitions, visiting, last_rm, last_ms)?;
            Some(last_offset.saturating_add(last_size))
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
///
/// When `is_row_major` is true, matrices use a virtual row-vector alignment
/// (based on the number of columns) instead of column-vector alignment.
fn type_alignment(
    ty: TypeId,
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    visiting: &mut HashSet<TypeId>,
    scalar_layout: bool,
    extended_alignment: bool,
    is_row_major: bool,
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
            let elem_align = type_alignment(
                elem,
                module,
                definitions,
                visiting,
                scalar_layout,
                extended_alignment,
                false,
            )?;
            if scalar_layout {
                Some(elem_align)
            } else {
                // vec2 aligns to 2N, vec3/vec4 align to 4N (where N is scalar alignment)
                let multiplier = if count == 2 { 2 } else { 4 };
                elem_align.checked_mul(multiplier)
            }
        }
        Op::TypeMatrix => {
            let (column, num_columns) = matrix_info(inst);
            let (column, num_columns) = (column?, num_columns?);
            let base_align = if is_row_major && !scalar_layout {
                // Row-major: alignment of a virtual vector of num_columns scalar
                // components (C++ getBaseAlignment, validate_decorations.cpp:210-219).
                let col_inst =
                    definitions.get(&ResultId::try_from(u32::from(column)).ok()?)?;
                let (scalar_type, _) = vector_info(col_inst);
                let scalar_type = scalar_type?;
                let scalar_align = type_alignment(
                    scalar_type,
                    module,
                    definitions,
                    visiting,
                    scalar_layout,
                    extended_alignment,
                    false,
                )?;
                let multiplier = if num_columns == 2 { 2 } else { 4 };
                scalar_align.checked_mul(multiplier)?
            } else {
                // Column-major (or scalar layout): alignment of column vector.
                type_alignment(
                    column,
                    module,
                    definitions,
                    visiting,
                    scalar_layout,
                    extended_alignment,
                    false,
                )?
            };
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
            // Propagate is_row_major through arrays (array of matrices inherits
            // majorness from the struct member decoration).
            let base_align = type_alignment(
                elem,
                module,
                definitions,
                visiting,
                scalar_layout,
                extended_alignment,
                is_row_major,
            )?;
            if extended_alignment && !scalar_layout {
                Some(round_up(base_align, 16))
            } else {
                Some(base_align)
            }
        }
        Op::TypeStruct => {
            let struct_id = ResultId::try_from(u32::from(ty)).ok()?;
            let mut max_align = 1;
            for (idx, op) in inst.operands.iter().enumerate() {
                let member_ty = match op {
                    rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok()?,
                    _ => return None,
                };
                // Each struct member has its own RowMajor/ColMajor decoration.
                let member_rm =
                    member_is_row_major(module, struct_id, MemberIndex(idx as u32));
                let align = type_alignment(
                    member_ty,
                    module,
                    definitions,
                    visiting,
                    scalar_layout,
                    extended_alignment,
                    member_rm,
                )?;
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
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u32> {
    let elem = vector_inst.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::IdRef(id) => TypeId::try_from(*id).ok(),
        _ => None,
    })?;
    type_alignment(elem, module, definitions, &mut HashSet::new(), true, false, false)
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

fn member_has_decoration(
    module: &Module,
    struct_id: ResultId,
    member: MemberIndex,
    decoration: Decoration,
) -> bool {
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
        let Some(rspirv::dr::Operand::Decoration(dec)) = ops.next() else {
            continue;
        };
        if *dec == decoration {
            return true;
        }
    }
    false
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
// Pre-layout Decoration Checks
// ============================================================================

/// Validates that all required decorations are present on a block struct before
/// running layout checks. This matches the C++ `checkForRequiredDecoration`
/// calls in CheckDecorationsOfBuffers (lines 1439-1476):
/// - All OpTypeArray members must have ArrayStride decorations
/// - All OpTypeMatrix members (including those inside arrays) must have
///   MatrixStride decorations
/// - All OpTypeMatrix members must have RowMajor or ColMajor decorations
fn check_required_block_decorations(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    struct_id: ResultId,
    deco_name: &str,
    visiting: &mut HashSet<ResultId>,
) -> ValidationResult {
    if !visiting.insert(struct_id) {
        return Ok(());
    }
    let Some(struct_inst) = definitions.get(&struct_id) else {
        visiting.remove(&struct_id);
        return Ok(());
    };
    if struct_inst.class.opcode != Op::TypeStruct {
        visiting.remove(&struct_id);
        return Ok(());
    }

    // Check that all members have Offset decorations (C++ isMissingOffsetInStruct, line 1432).
    let offsets = collect_member_offsets(module, struct_id);
    for idx in 0..struct_inst.operands.len() {
        if !offsets.contains_key(&MemberIndex(idx as u32)) {
            visiting.remove(&struct_id);
            return Err(ValidationError::InvalidBlockLayout {
                struct_type: struct_id,
                reason: format!(
                    "Structure decorated as {} must be explicitly laid out with Offset decorations",
                    deco_name
                ),
            }
            .into());
        }
    }

    for (member_idx, op) in struct_inst.operands.iter().enumerate() {
        let rspirv::dr::Operand::IdRef(member_type_raw) = op else {
            continue;
        };
        let Ok(member_type_id) = ResultId::try_from(*member_type_raw) else {
            continue;
        };
        let Some(member_inst) = definitions.get(&member_type_id) else {
            continue;
        };

        // ArrayStride: direct array members must have ArrayStride
        if member_inst.class.opcode == Op::TypeArray {
            if array_stride(module, member_type_id).is_none() {
                visiting.remove(&struct_id);
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "Structure decorated as {} must be explicitly laid out with ArrayStride decorations",
                        deco_name
                    ),
                }
                .into());
            }
        }

        // For matrix checks, unwrap arrays to find the underlying type
        let mut effective_inst = member_inst;
        while effective_inst.class.opcode == Op::TypeArray
            || effective_inst.class.opcode == Op::TypeRuntimeArray
        {
            let Some(rspirv::dr::Operand::IdRef(e)) = effective_inst.operands.first() else {
                break;
            };
            let Ok(eid) = ResultId::try_from(*e) else {
                break;
            };
            let Some(einst) = definitions.get(&eid) else {
                break;
            };
            effective_inst = einst;
        }

        if effective_inst.class.opcode == Op::TypeMatrix {
            let midx = MemberIndex(member_idx as u32);
            // MatrixStride must be present
            if member_matrix_stride(module, struct_id, midx).is_none() {
                visiting.remove(&struct_id);
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "Structure decorated as {} must be explicitly laid out with MatrixStride decorations",
                        deco_name
                    ),
                }
                .into());
            }
            // RowMajor or ColMajor must be present
            if !member_has_decoration(module, struct_id, midx, Decoration::RowMajor)
                && !member_has_decoration(module, struct_id, midx, Decoration::ColMajor)
            {
                visiting.remove(&struct_id);
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "Structure decorated as {} must be explicitly laid out with RowMajor or ColMajor decorations",
                        deco_name
                    ),
                }
                .into());
            }
        }

        // Recurse into nested struct members
        if member_inst.class.opcode == Op::TypeStruct {
            check_required_block_decorations(
                module,
                definitions,
                member_type_id,
                deco_name,
                visiting,
            )?;
        }
    }

    visiting.remove(&struct_id);
    Ok(())
}

// ============================================================================
// Recursive Struct Layout Checker
// ============================================================================

/// Validates the layout of a single block struct, recursively checking nested
/// structs and arrays. This matches the C++ `checkLayout` function in
/// validate_decorations.cpp.
fn check_struct_layout(
    module: &Module,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
    struct_id: ResultId,
    incoming_offset: u32,
    scalar_layout: bool,
    extended_alignment: bool,
    relax_block_layout: bool,
    depth: u32,
) -> ValidationResult {
    // Depth limit to prevent infinite recursion from circular types.
    if depth > 100 {
        return Ok(());
    }

    let Some(struct_inst) = definitions.get(&struct_id) else {
        return Ok(());
    };
    if struct_inst.class.opcode != Op::TypeStruct || struct_inst.operands.is_empty() {
        return Ok(());
    }

    let member_offsets_map = collect_member_offsets(module, struct_id);

    // Build member list sorted by absolute offset (matching C++ checkLayout).
    let mut sorted_members: Vec<(u32, u32)> = Vec::with_capacity(struct_inst.operands.len());
    for index in 0..struct_inst.operands.len() {
        let Some(&off) = member_offsets_map.get(&MemberIndex(index as u32)) else {
            return Err(ValidationError::InvalidBlockLayout {
                struct_type: struct_id,
                reason: "missing OpMemberDecorate Offset".to_string(),
            }
            .into());
        };
        sorted_members.push((index as u32, incoming_offset.saturating_add(off)));
    }
    sorted_members.sort_by_key(|&(_, offset)| offset);

    let mut next_valid_offset: u32 = 0;

    for (ordered_idx, &(member_idx, offset)) in sorted_members.iter().enumerate() {
        let rspirv::dr::Operand::IdRef(member_type_id_raw) =
            &struct_inst.operands[member_idx as usize]
        else {
            continue;
        };
        let Ok(member_type_id) = TypeId::try_from(*member_type_id_raw) else {
            continue;
        };
        let Ok(member_result_id) = ResultId::try_from(u32::from(member_type_id)) else {
            continue;
        };
        let Some(member_inst) = definitions.get(&member_result_id) else {
            continue;
        };

        // Look up matrix majorness and stride for this member (inherited
        // through arrays to contained matrices, matching C++ LayoutConstraints).
        let is_row_major = member_is_row_major(module, struct_id, MemberIndex(member_idx));
        let mat_stride =
            member_matrix_stride(module, struct_id, MemberIndex(member_idx)).unwrap_or(0);

        let Some(alignment) = type_alignment(
            member_type_id,
            module,
            definitions,
            &mut HashSet::new(),
            scalar_layout,
            extended_alignment,
            is_row_major,
        ) else {
            continue;
        };

        // Runtime arrays have unknown size; use 0 (matching C++ getSize).
        let size = if member_inst.class.opcode == Op::TypeRuntimeArray {
            0
        } else {
            match type_layout_size(
                member_type_id,
                module,
                definitions,
                &mut HashSet::new(),
                is_row_major,
                mat_stride,
            ) {
                Some(s) => s,
                None => continue,
            }
        };

        // Runtime array must be the last member by offset order (C++ line 573).
        if member_inst.class.opcode == Op::TypeRuntimeArray
            && ordered_idx + 1 != sorted_members.len()
        {
            return Err(ValidationError::InvalidBlockLayout {
                struct_type: struct_id,
                reason: "runtime array member must be the final struct member".to_string(),
            }
            .into());
        }

        // Offset alignment checks.
        if relax_block_layout && !scalar_layout && member_inst.class.opcode == Op::TypeVector {
            // Relaxed layout: vector offset aligned to scalar element alignment.
            let Some(scalar_align) = vector_scalar_alignment(member_inst, module, definitions) else {
                continue;
            };
            if offset % scalar_align != 0 {
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "member offset {} is not aligned to vector scalar element size {}",
                        offset, scalar_align
                    ),
                }
                .into());
            }
        } else if offset % alignment != 0 {
            return Err(ValidationError::InvalidBlockLayout {
                struct_type: struct_id,
                reason: format!(
                    "member offset {} is not aligned to required alignment {}",
                    offset, alignment
                ),
            }
            .into());
        }

        // Overlap/padding check (C++ line 601).
        if offset < next_valid_offset {
            return Err(ValidationError::InvalidBlockLayout {
                struct_type: struct_id,
                reason: "member offsets overlap".to_string(),
            }
            .into());
        }

        // Vector straddle check under relaxed, non-scalar layout (C++ line 605).
        if relax_block_layout && !scalar_layout && member_inst.class.opcode == Op::TypeVector {
            let straddles = if size <= 16 {
                size > 0 && (offset >> 4) != ((offset.saturating_add(size - 1)) >> 4)
            } else {
                offset % 16 != 0
            };
            if straddles {
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!(
                        "vector at offset {} improperly straddles a 16-byte boundary",
                        offset
                    ),
                }
                .into());
            }
        }

        // Recursive check for nested structs (C++ line 614).
        if member_inst.class.opcode == Op::TypeStruct {
            check_struct_layout(
                module,
                definitions,
                member_result_id,
                offset,
                scalar_layout,
                extended_alignment,
                relax_block_layout,
                depth + 1,
            )?;
        }

        // Matrix stride check (C++ line 622).
        if member_inst.class.opcode == Op::TypeMatrix {
            let stride =
                member_matrix_stride(module, struct_id, MemberIndex(member_idx)).ok_or_else(
                    || -> SpannedValidationError {
                        ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: "matrix member is missing MatrixStride".to_string(),
                        }
                        .into()
                    },
                )?;
            if stride % alignment != 0 {
                return Err(ValidationError::InvalidBlockLayout {
                    struct_type: struct_id,
                    reason: format!("matrix stride {} is not aligned to {}", stride, alignment),
                }
                .into());
            }
            let (column_type, _) = matrix_info(member_inst);
            if let Some(col_ty) = column_type {
                if let Some(col_size) =
                    type_layout_size(col_ty, module, definitions, &mut HashSet::new(), false, 0)
                {
                    if col_size > stride {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "matrix stride {} is smaller than column size {}",
                                stride, col_size
                            ),
                        }
                        .into());
                    }
                }
            }
        }

        // Check arrays and runtime arrays recursively (C++ lines 631-707).
        let mut array_inst = member_inst;
        let mut array_result_id = member_result_id;
        let mut array_alignment = alignment;
        while array_inst.class.opcode == Op::TypeArray
            || array_inst.class.opcode == Op::TypeRuntimeArray
        {
            let Some(rspirv::dr::Operand::IdRef(elem_raw)) = array_inst.operands.first() else {
                break;
            };
            let Ok(elem_type) = TypeId::try_from(*elem_raw) else {
                break;
            };
            let Ok(elem_result_id) = ResultId::try_from(u32::from(elem_type)) else {
                break;
            };
            let Some(elem_inst) = definitions.get(&elem_result_id) else {
                break;
            };

            // Check array stride (C++ lines 638-652).
            let stride = array_stride(module, array_result_id);
            if let Some(s) = stride {
                if s == 0 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: "array has stride 0".to_string(),
                    }
                    .into());
                }
                if s % array_alignment != 0 {
                    return Err(ValidationError::InvalidBlockLayout {
                        struct_type: struct_id,
                        reason: format!(
                            "array stride {} is not aligned to {}",
                            s, array_alignment
                        ),
                    }
                    .into());
                }
            }
            let stride_val = stride.unwrap_or(0);

            let num_elements = if array_inst.class.opcode == Op::TypeArray {
                array_length(array_inst, definitions).unwrap_or(1).max(1)
            } else {
                1
            };

            // If element is struct, recursively validate at each array offset
            // (C++ lines 666-682). Stop when offsets repeat mod 16.
            if elem_inst.class.opcode == Op::TypeStruct && stride_val > 0 {
                let mut seen = [false; 16];
                for i in 0..num_elements {
                    let next_offset = offset.saturating_add(i.saturating_mul(stride_val));
                    let bucket = (next_offset % 16) as usize;
                    if seen[bucket] {
                        break;
                    }
                    check_struct_layout(
                        module,
                        definitions,
                        elem_result_id,
                        next_offset,
                        scalar_layout,
                        extended_alignment,
                        relax_block_layout,
                        depth + 1,
                    )?;
                    seen[bucket] = true;
                }
            } else if elem_inst.class.opcode == Op::TypeMatrix {
                // Matrix stride for matrices inside arrays (C++ lines 683-691).
                if let Some(ms) =
                    member_matrix_stride(module, struct_id, MemberIndex(member_idx))
                {
                    if ms % alignment != 0 {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "matrix stride {} in array is not aligned to {}",
                                ms, alignment
                            ),
                        }
                        .into());
                    }
                }
            }

            // Check element_size <= stride (C++ lines 700-706).
            if stride_val > 0 {
                if let Some(element_size) = type_layout_size(
                    elem_type,
                    module,
                    definitions,
                    &mut HashSet::new(),
                    is_row_major,
                    mat_stride,
                ) {
                    if element_size > stride_val {
                        return Err(ValidationError::InvalidBlockLayout {
                            struct_type: struct_id,
                            reason: format!(
                                "array stride {} is smaller than element size {}",
                                stride_val, element_size
                            ),
                        }
                        .into());
                    }
                }
            }

            // Descend to element type (C++ lines 694-698).
            array_inst = elem_inst;
            array_result_id = elem_result_id;
            array_alignment = type_alignment(
                elem_type,
                module,
                definitions,
                &mut HashSet::new(),
                scalar_layout,
                extended_alignment,
                is_row_major,
            )
            .unwrap_or(1);
        }

        // Update next valid offset (C++ lines 708-714).
        next_valid_offset = offset.saturating_add(size);
        if !scalar_layout
            && matches!(
                member_inst.class.opcode,
                Op::TypeArray | Op::TypeStruct
            )
        {
            next_valid_offset = round_up(next_valid_offset, alignment);
        }
    }

    Ok(())
}

// ============================================================================
// All block layout rules
// ============================================================================

/// Returns all block layout validation rules.
pub fn all_block_layout_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&BlockLayoutRule]
}
