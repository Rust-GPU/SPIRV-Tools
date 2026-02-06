//! Image instruction validation rules.
//!
//! This module validates SPIR-V image instructions including:
//!
//! - OpTypeImage and OpTypeSampledImage structure
//! - Image sampling operations (ImplicitLod, ExplicitLod)
//! - Image read/write operations
//! - Image gather operations
//! - Image fetch operations
//! - Image query operations
//! - Image operand masks (Bias, Lod, Grad, Offset, etc.)
//! - Sparse image operations
//! - OpSampledImage same-block consumer validation

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{
    Capability, Decoration, Dim, ExecutionMode, ExecutionModel, ImageFormat, ImageOperands, Op,
};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::op_ext::OpExt;
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};
use crate::validation::ValidationResult;

// ============================================================================
// Image Type Information
// ============================================================================

/// Information extracted from OpTypeImage.
#[derive(Debug, Clone)]
pub struct ImageTypeInfo {
    /// The sampled type (component type)
    pub sampled_type: u32,
    /// Image dimensionality
    pub dim: Dim,
    /// Depth image flag (0=not depth, 1=depth, 2=no indication)
    pub depth: u32,
    /// Array flag (0=non-arrayed, 1=arrayed)
    pub arrayed: u32,
    /// Multisampled flag (0=single-sampled, 1=multisampled)
    pub multisampled: u32,
    /// Sampled flag (0=runtime, 1=sampling, 2=read/write)
    pub sampled: u32,
    /// Image format
    pub format: ImageFormat,
}

impl Default for ImageTypeInfo {
    fn default() -> Self {
        Self {
            sampled_type: 0,
            dim: Dim::Dim1D,
            depth: 0,
            arrayed: 0,
            multisampled: 0,
            sampled: 0,
            format: ImageFormat::Unknown,
        }
    }
}

impl ImageTypeInfo {
    /// Extract image type info from an OpTypeImage or OpTypeSampledImage instruction.
    pub fn from_type_id(type_id: u32, ctx: &ValidationContext<'_>) -> Option<Self> {
        let result_id = ResultId::try_from(type_id).ok()?;
        let inst = ctx.definitions.get(&result_id)?;

        // Handle OpTypeSampledImage - extract inner image type
        let image_inst = if inst.class.opcode == Op::TypeSampledImage {
            let inner_type_id = inst.operands.first()?.id_ref_any()?;
            let inner_result_id = ResultId::try_from(inner_type_id).ok()?;
            ctx.definitions.get(&inner_result_id)?
        } else {
            inst
        };

        if image_inst.class.opcode != Op::TypeImage {
            return None;
        }

        // OpTypeImage: sampled_type, dim, depth, arrayed, ms, sampled, format, [access]
        if image_inst.operands.len() < 7 {
            return None;
        }

        Some(ImageTypeInfo {
            sampled_type: image_inst.operands.first()?.id_ref_any()?,
            dim: match &image_inst.operands.get(1)? {
                Operand::Dim(d) => *d,
                _ => return None,
            },
            depth: match &image_inst.operands.get(2)? {
                Operand::LiteralBit32(v) => *v,
                _ => return None,
            },
            arrayed: match &image_inst.operands.get(3)? {
                Operand::LiteralBit32(v) => *v,
                _ => return None,
            },
            multisampled: match &image_inst.operands.get(4)? {
                Operand::LiteralBit32(v) => *v,
                _ => return None,
            },
            sampled: match &image_inst.operands.get(5)? {
                Operand::LiteralBit32(v) => *v,
                _ => return None,
            },
            format: match &image_inst.operands.get(6)? {
                Operand::ImageFormat(f) => *f,
                _ => return None,
            },
        })
    }
}

// Note: Image opcode classification is provided by the OpExt trait from op_ext module.

// ============================================================================
// Coordinate Size Calculation
// ============================================================================

/// Get the number of coordinate components for a single plane.
fn get_plane_coord_size(info: &ImageTypeInfo) -> u32 {
    match info.dim {
        Dim::Dim1D | Dim::DimBuffer => 1,
        Dim::Dim2D | Dim::DimRect | Dim::DimSubpassData | Dim::DimTileImageDataEXT => 2,
        Dim::Dim3D | Dim::DimCube => 3,
    }
}

/// Get the minimum coordinate size for an image operation.
fn get_min_coord_size(op: Op, info: &ImageTypeInfo) -> u32 {
    // Read/Write on Cube use UV (2D), not direction vector
    if info.dim == Dim::DimCube && op.is_image_read_write() {
        return 3;
    }
    get_plane_coord_size(info) + info.arrayed + if op.is_proj() { 1 } else { 0 }
}

// ============================================================================
// Image Type Rule
// ============================================================================

/// Validates OpTypeImage structure.
pub struct ImageTypeRule;

impl ValidationRule for ImageTypeRule {
    fn name(&self) -> &'static str {
        "image-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeImage {
                continue;
            }

            let type_id = inst.result_id.and_then(|id| TypeId::try_from(id).ok());

            // Validate operand count
            if inst.operands.len() < 7 {
                return Err(ValidationError::ImageTypeInvalidOperandCount {
                    type_id,
                    expected: 7,
                    actual: inst.operands.len(),
                }
                .into());
            }

            // Extract Sampled Type (operand 0)
            let sampled_type_id = match &inst.operands.get(0) {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            };

            // Validate Sampled Type is numeric scalar or void
            if let Some(st_id) = sampled_type_id {
                if let Ok(rid) = crate::validation::types::ResultId::try_from(st_id) {
                    if let Some(st_inst) = ctx.definitions.get(&rid) {
                        let valid = matches!(
                            st_inst.class.opcode,
                            Op::TypeVoid | Op::TypeInt | Op::TypeFloat
                        );
                        if !valid {
                            return Err(
                                ValidationError::ImageTypeInvalidSampledType { type_id }.into()
                            );
                        }

                        // Int64ImageEXT capability check: 64-bit int sampled type requires it
                        if st_inst.class.opcode == Op::TypeInt {
                            if let Some(Operand::LiteralBit32(width)) = st_inst.operands.first() {
                                if *width == 64 && !ctx.has_capability(Capability::Int64ImageEXT) {
                                    return Err(
                                        ValidationError::ImageTypeRequiresInt64ImageCapability
                                            .into(),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Extract Dim
            let dim = match &inst.operands.get(1) {
                Some(Operand::Dim(d)) => *d,
                _ => continue,
            };

            // Extract Depth (operand 2)
            let depth = match &inst.operands.get(2) {
                Some(Operand::LiteralBit32(v)) => *v,
                _ => continue,
            };

            // Extract Arrayed (operand 3)
            let arrayed = match &inst.operands.get(3) {
                Some(Operand::LiteralBit32(v)) => *v,
                _ => continue,
            };

            // Extract MS (operand 4)
            let ms = match &inst.operands.get(4) {
                Some(Operand::LiteralBit32(v)) => *v,
                _ => continue,
            };

            // Extract Sampled (operand 5)
            let sampled = match &inst.operands.get(5) {
                Some(Operand::LiteralBit32(v)) => *v,
                _ => continue,
            };

            // Extract Format (operand 6)
            let format = match &inst.operands.get(6) {
                Some(Operand::ImageFormat(f)) => *f,
                _ => continue,
            };

            // Validate Depth must be 0, 1, or 2
            if depth > 2 {
                return Err(ValidationError::ImageTypeInvalidDepthValue {
                    type_id,
                    value: depth,
                }
                .into());
            }

            // Validate Arrayed must be 0 or 1
            if arrayed > 1 {
                return Err(ValidationError::ImageTypeInvalidArrayedValue {
                    type_id,
                    value: arrayed,
                }
                .into());
            }

            // Validate MS must be 0 or 1
            if ms > 1 {
                return Err(ValidationError::ImageTypeInvalidMsValue { type_id, value: ms }.into());
            }

            // Validate Sampled must be 0, 1, or 2
            if sampled > 2 {
                return Err(ValidationError::ImageTypeInvalidSampledValue {
                    type_id,
                    value: sampled,
                }
                .into());
            }

            // StorageImageMultisample: multisampled storage images require the capability
            // (except for TileImageDataEXT dimension)
            if dim != Dim::DimTileImageDataEXT
                && ms != 0
                && sampled == 2
                && !ctx.has_capability(Capability::StorageImageMultisample)
            {
                return Err(
                    ValidationError::ImageTypeRequiresStorageImageMultisampleCapability.into(),
                );
            }

            // Vulkan: Sampled must be 1 or 2 (cannot be 0)
            if ctx.env.is_vulkan() && sampled == 0 {
                return Err(
                    ValidationError::ImageTypeSampledMustBeOneOrTwoInVulkan { type_id }.into(),
                );
            }

            // SubpassData requires Vulkan environment
            if dim == Dim::DimSubpassData && !ctx.env.is_vulkan() {
                return Err(ValidationError::ImageTypeSubpassDataRequiresVulkan {
                    type_id,
                    env: ctx.env,
                }
                .into());
            }

            // SubpassData constraints
            if dim == Dim::DimSubpassData {
                if arrayed != 0 {
                    return Err(
                        ValidationError::ImageTypeSubpassDataMustNotBeArrayed { type_id }.into(),
                    );
                }
                if sampled != 2 {
                    return Err(
                        ValidationError::ImageTypeSubpassDataSampledMustBeTwo { type_id }.into(),
                    );
                }
                if format != rspirv::spirv::ImageFormat::Unknown {
                    return Err(ValidationError::ImageTypeSubpassDataFormatMustBeUnknown {
                        type_id,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Operand Validation Rule
// ============================================================================

/// Validates image operand masks and their parameters.
pub struct ImageOperandRule;

impl ValidationRule for ImageOperandRule {
    fn name(&self) -> &'static str {
        "image-operand"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !inst.class.opcode.is_image_op() {
                        continue;
                    }

                    // Skip query operations - they don't have image operands
                    if inst.class.opcode.is_image_query() {
                        continue;
                    }

                    // Get image type info
                    let image_type_info = get_image_type_from_instruction(inst, ctx);

                    // Find the image operand mask in the instruction
                    let mask_result = find_image_operand_mask(inst);
                    let (mask, _operand_start_idx) = match mask_result {
                        Some(m) => m,
                        None => continue,
                    };

                    // Validate multisampled images require Sample operand for fetch, read, and write operations
                    // Query operations (OpImageQuerySize, etc.) don't require the Sample operand
                    if let Some(ref info) = image_type_info {
                        let requires_sample =
                            inst.class.opcode.is_fetch() || inst.class.opcode.is_image_read_write();
                        if info.multisampled != 0
                            && requires_sample
                            && !mask.contains(ImageOperands::SAMPLE)
                        {
                            return Err(
                                ValidationError::ImageOperandSampleRequiredForMultisampled {
                                    function: function_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into(),
                            );
                        }
                    }

                    // Validate mutually exclusive offset operands
                    let offset_count = [
                        mask.contains(ImageOperands::OFFSET),
                        mask.contains(ImageOperands::CONST_OFFSET),
                        mask.contains(ImageOperands::CONST_OFFSETS),
                    ]
                    .iter()
                    .filter(|&&x| x)
                    .count();

                    if offset_count > 1 {
                        return Err(ValidationError::ImageOperandMultipleOffsets {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        }
                        .into());
                    }

                    // Validate Bias operand
                    if mask.contains(ImageOperands::BIAS)
                        && !inst.class.opcode.is_implicit_lod()
                        && !(inst.class.opcode.is_gather()
                            && ctx.has_capability(Capability::ImageGatherBiasLodAMD))
                    {
                        return Err(ValidationError::ImageOperandBiasRequiresImplicitLod {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        }
                        .into());
                    }

                    // Validate Lod operand
                    if mask.contains(ImageOperands::LOD) {
                        let valid_for_lod = inst.class.opcode.is_explicit_lod()
                            || inst.class.opcode.is_fetch()
                            || (inst.class.opcode.is_image_read_write()
                                && ctx.has_capability(Capability::ImageReadWriteLodAMD))
                            || (inst.class.opcode.is_gather()
                                && ctx.has_capability(Capability::ImageGatherBiasLodAMD));

                        if !valid_for_lod {
                            return Err(
                                ValidationError::ImageOperandLodRequiresExplicitLodOrFetch {
                                    function: function_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                }
                                .into(),
                            );
                        }

                        // Lod and Grad are mutually exclusive
                        if mask.contains(ImageOperands::GRAD) {
                            return Err(ValidationError::ImageOperandLodAndGradMutuallyExclusive {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // Validate Grad operand
                    if mask.contains(ImageOperands::GRAD) && !inst.class.opcode.is_explicit_lod() {
                        return Err(ValidationError::ImageOperandGradRequiresExplicitLod {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        }
                        .into());
                    }

                    // Validate ConstOffsets operand
                    if mask.contains(ImageOperands::CONST_OFFSETS) && !inst.class.opcode.is_gather()
                    {
                        return Err(ValidationError::ImageOperandConstOffsetsRequiresGather {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        }
                        .into());
                    }

                    // Validate offset operands cannot be used with Cube
                    if let Some(ref info) = image_type_info {
                        if info.dim == Dim::DimCube
                            && (mask.contains(ImageOperands::OFFSET)
                                || mask.contains(ImageOperands::CONST_OFFSET)
                                || mask.contains(ImageOperands::CONST_OFFSETS))
                        {
                            return Err(ValidationError::ImageOperandOffsetCannotBeUsedWithCube {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }

                    // MakeTexelAvailable can only be used with OpImageWrite
                    if mask.contains(ImageOperands::MAKE_TEXEL_AVAILABLE)
                        && inst.class.opcode != Op::ImageWrite
                    {
                        return Err(
                            ValidationError::ImageOperandMakeTexelAvailableRequiresWrite {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into(),
                        );
                    }

                    // MakeTexelVisible cannot be used with OpImageWrite
                    if mask.contains(ImageOperands::MAKE_TEXEL_VISIBLE)
                        && inst.class.opcode == Op::ImageWrite
                    {
                        return Err(
                            ValidationError::ImageOperandMakeTexelVisibleCannotBeUsedWithWrite {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into(),
                        );
                    }

                    // MakeTexelAvailable requires NonPrivateTexel
                    if mask.contains(ImageOperands::MAKE_TEXEL_AVAILABLE)
                        && !mask.contains(ImageOperands::NON_PRIVATE_TEXEL)
                    {
                        return Err(
                            ValidationError::ImageOperandMakeTexelAvailableRequiresNonPrivate {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into(),
                        );
                    }

                    // MakeTexelVisible requires NonPrivateTexel
                    if mask.contains(ImageOperands::MAKE_TEXEL_VISIBLE)
                        && !mask.contains(ImageOperands::NON_PRIVATE_TEXEL)
                    {
                        return Err(
                            ValidationError::ImageOperandMakeTexelVisibleRequiresNonPrivate {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into(),
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Operand Type Rule
// ============================================================================

/// Gets the type instruction for a value operand ID.
/// Given an ID, looks up its defining instruction, gets its result_type,
/// and returns the type instruction.
fn get_value_type_id(operand_id: u32, definitions: &HashMap<ResultId, Instruction>) -> Option<u32> {
    let rid = ResultId::try_from(operand_id).ok()?;
    let inst = definitions.get(&rid)?;
    inst.result_type
}

/// Returns true if the defining instruction for an ID is a constant opcode.
fn is_constant_id(id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    let Some(rid) = ResultId::try_from(id).ok() else {
        return false;
    };
    let Some(inst) = definitions.get(&rid) else {
        return false;
    };
    matches!(
        inst.class.opcode,
        Op::Constant
            | Op::ConstantComposite
            | Op::ConstantNull
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstantOp
    )
}

/// Validates the types of image operand values (Bias, Lod, Grad, etc.).
pub struct ImageOperandTypeRule;

impl ValidationRule for ImageOperandTypeRule {
    fn name(&self) -> &'static str {
        "image-operand-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !inst.class.opcode.is_image_op() || inst.class.opcode.is_image_query() {
                        continue;
                    }

                    let image_type_info = get_image_type_from_instruction(inst, ctx);

                    let Some((mask, operand_start_idx)) = find_image_operand_mask(inst) else {
                        continue;
                    };

                    if mask.is_empty() {
                        continue;
                    }

                    let opcode = inst.class.opcode;
                    let mut word_idx = operand_start_idx;

                    // Walk through flags in bit order and type-check each dependent operand.

                    // Bias (bit 0)
                    if mask.contains(ImageOperands::BIAS) {
                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            word_idx += 1;
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                if !resolver.is_float_scalar(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(
                                        ValidationError::ImageOperandBiasNotFloat32Scalar {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        } else {
                            word_idx += 1;
                        }

                        if let Some(ref info) = image_type_info {
                            if !matches!(
                                info.dim,
                                Dim::Dim1D | Dim::Dim2D | Dim::Dim3D | Dim::DimCube
                            ) {
                                return Err(ValidationError::ImageOperandBiasInvalidDim {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Lod (bit 1)
                    if mask.contains(ImageOperands::LOD) {
                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            word_idx += 1;
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                let is_explicit = opcode.is_explicit_lod();
                                let is_gather_lod_bias_amd = opcode.is_gather()
                                    && ctx.has_capability(Capability::ImageGatherBiasLodAMD);

                                if is_explicit || is_gather_lod_bias_amd {
                                    // Must be 32-bit float scalar
                                    if !resolver.is_float_scalar(type_id, &ctx.definitions)
                                        || resolver.get_bit_width(type_id, &ctx.definitions)
                                            != Some(32)
                                    {
                                        return Err(
                                            ValidationError::ImageOperandLodNotFloat32ScalarForExplicit {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                            }
                                            .into(),
                                        );
                                    }
                                } else {
                                    // Must be 32-bit int scalar (for Fetch)
                                    if !resolver.is_int_scalar(type_id, &ctx.definitions)
                                        || resolver.get_bit_width(type_id, &ctx.definitions)
                                            != Some(32)
                                    {
                                        return Err(
                                            ValidationError::ImageOperandLodNotInt32ScalarForFetch {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }
                        } else {
                            word_idx += 1;
                        }

                        if let Some(ref info) = image_type_info {
                            if !matches!(
                                info.dim,
                                Dim::Dim1D | Dim::Dim2D | Dim::Dim3D | Dim::DimCube
                            ) {
                                return Err(ValidationError::ImageOperandLodInvalidDim {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                            if info.multisampled != 0 {
                                return Err(ValidationError::ImageOperandLodRequiresMsZero {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }
                    }

                    // Grad (bit 2) - consumes TWO operands (dx, dy)
                    if mask.contains(ImageOperands::GRAD) {
                        let dx_id = inst.operands.get(word_idx).and_then(|o| o.id_ref_any());
                        word_idx += 1;
                        let dy_id = inst.operands.get(word_idx).and_then(|o| o.id_ref_any());
                        word_idx += 1;

                        if let (Some(dx_id), Some(dy_id)) = (dx_id, dy_id) {
                            let dx_type = get_value_type_id(dx_id, &ctx.definitions);
                            let dy_type = get_value_type_id(dy_id, &ctx.definitions);

                            // Both must be 32-bit float scalar or vector
                            for type_id in [dx_type, dy_type].into_iter().flatten() {
                                if !resolver.is_float_scalar_or_vector(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(ValidationError::ImageOperandGradNotFloat32 {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                    }
                                    .into());
                                }
                            }

                            // Component count must match plane coord size
                            if let Some(ref info) = image_type_info {
                                let plane_size = get_plane_coord_size(info);
                                for type_id in [dx_type, dy_type].into_iter().flatten() {
                                    let dim = resolver.get_dimension(type_id, &ctx.definitions);
                                    if plane_size != dim {
                                        return Err(
                                            ValidationError::ImageOperandGradComponentCountMismatch {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                                expected: plane_size,
                                                actual: dim,
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // ConstOffset (bit 3)
                    if mask.contains(ImageOperands::CONST_OFFSET) {
                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            word_idx += 1;
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                if !resolver.is_int_scalar_or_vector(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(ValidationError::ImageOperandOffsetNotInt32 {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        operand_name: "ConstOffset",
                                    }
                                    .into());
                                }

                                if let Some(ref info) = image_type_info {
                                    let plane_size = get_plane_coord_size(info);
                                    let dim = resolver.get_dimension(type_id, &ctx.definitions);
                                    if plane_size != dim {
                                        return Err(
                                            ValidationError::ImageOperandOffsetComponentCountMismatch {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                                operand_name: "ConstOffset",
                                                expected: plane_size,
                                                actual: dim,
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }

                            // Must be a constant
                            if !is_constant_id(operand_id, &ctx.definitions) {
                                return Err(ValidationError::ImageOperandConstOffsetNotConstant {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        } else {
                            word_idx += 1;
                        }
                    }

                    // Offset (bit 4)
                    if mask.contains(ImageOperands::OFFSET) {
                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            word_idx += 1;
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                if !resolver.is_int_scalar_or_vector(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(ValidationError::ImageOperandOffsetNotInt32 {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        operand_name: "Offset",
                                    }
                                    .into());
                                }

                                if let Some(ref info) = image_type_info {
                                    let plane_size = get_plane_coord_size(info);
                                    let dim = resolver.get_dimension(type_id, &ctx.definitions);
                                    if plane_size != dim {
                                        return Err(
                                            ValidationError::ImageOperandOffsetComponentCountMismatch {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                                operand_name: "Offset",
                                                expected: plane_size,
                                                actual: dim,
                                            }
                                            .into(),
                                        );
                                    }
                                }
                            }
                        } else {
                            word_idx += 1;
                        }
                    }

                    // ConstOffsets (bit 5) - skip detailed validation for now, just advance
                    if mask.contains(ImageOperands::CONST_OFFSETS) {
                        word_idx += 1;
                    }

                    // Sample (bit 6)
                    if mask.contains(ImageOperands::SAMPLE) {
                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            word_idx += 1;
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                if !resolver.is_int_scalar(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(
                                        ValidationError::ImageOperandSampleNotInt32Scalar {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        } else {
                            word_idx += 1;
                        }
                    }

                    // MinLod (bit 7)
                    if mask.contains(ImageOperands::MIN_LOD) {
                        // MinLod only valid with ImplicitLod or Grad
                        if !opcode.is_implicit_lod() && !mask.contains(ImageOperands::GRAD) {
                            return Err(
                                ValidationError::ImageOperandMinLodRequiresImplicitOrGrad {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into(),
                            );
                        }

                        if let Some(operand_id) =
                            inst.operands.get(word_idx).and_then(|o| o.id_ref_any())
                        {
                            let _ = word_idx; // last operand we check
                            if let Some(type_id) = get_value_type_id(operand_id, &ctx.definitions) {
                                if !resolver.is_float_scalar(type_id, &ctx.definitions)
                                    || resolver.get_bit_width(type_id, &ctx.definitions) != Some(32)
                                {
                                    return Err(
                                        ValidationError::ImageOperandMinLodNotFloat32Scalar {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }

                        if let Some(ref info) = image_type_info {
                            if !matches!(
                                info.dim,
                                Dim::Dim1D | Dim::Dim2D | Dim::Dim3D | Dim::DimCube
                            ) {
                                return Err(ValidationError::ImageOperandMinLodInvalidDim {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                            if info.multisampled != 0 {
                                return Err(ValidationError::ImageOperandMinLodRequiresMsZero {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Sample Execution Model Rule
// ============================================================================

/// Validates that implicit LOD operations are only used in Fragment shaders.
pub struct ImageSampleExecutionModelRule;

impl ValidationRule for ImageSampleExecutionModelRule {
    fn name(&self) -> &'static str {
        "image-sample-execution-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Check if we have implicit LOD operations
        let has_implicit_lod = ctx.module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions
                    .iter()
                    .any(|i| i.class.opcode.is_implicit_lod())
            })
        });

        if !has_implicit_lod {
            return Ok(());
        }

        // Implicit LOD is only valid in Fragment shader (or with derivative group execution modes)
        let has_fragment = ctx.entry_models.contains(&ExecutionModel::Fragment);
        let has_derivative_group = ctx.module.execution_modes.iter().any(|inst| {
            inst.operands.iter().any(|op| {
                matches!(
                    op,
                    rspirv::dr::Operand::ExecutionMode(
                        ExecutionMode::DerivativeGroupQuadsNV
                            | ExecutionMode::DerivativeGroupLinearNV
                    )
                )
            })
        });

        if !has_fragment && !has_derivative_group && !ctx.entry_models.is_empty() {
            // Find the offending instruction for error reporting
            for function in &ctx.module.functions {
                let function_id = function
                    .def
                    .as_ref()
                    .and_then(|d| d.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for block in &function.blocks {
                    let block_id = block
                        .label
                        .as_ref()
                        .and_then(|l| l.result_id)
                        .and_then(|id| Id::try_from(id).ok());

                    for inst in &block.instructions {
                        if inst.class.opcode.is_implicit_lod() {
                            return Err(ValidationError::ImageImplicitLodRequiresFragment {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Read/Write Rule
// ============================================================================

/// Validates image read/write operations.
pub struct ImageReadWriteRule;

impl ValidationRule for ImageReadWriteRule {
    fn name(&self) -> &'static str {
        "image-read-write"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if opcode == Op::ImageRead || opcode == Op::ImageSparseRead {
                        // Get image type info
                        if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                            // Image read requires sampled = 0 (runtime) or 2 (read/write)
                            if info.sampled == 1 {
                                return Err(ValidationError::ImageReadRequiresStorageImage {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }
                    }

                    if opcode == Op::ImageWrite {
                        // Get image type info
                        if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                            // Image write requires sampled = 0 (runtime) or 2 (read/write)
                            if info.sampled == 1 {
                                return Err(ValidationError::ImageWriteRequiresStorageImage {
                                    function: function_id,
                                    block: block_id,
                                }
                                .into());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Query Rule
// ============================================================================

/// Validates image query operations.
pub struct ImageQueryRule;

impl ValidationRule for ImageQueryRule {
    fn name(&self) -> &'static str {
        "image-query"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    match opcode {
                        Op::ImageQuerySizeLod => {
                            // Result must be int scalar or vector
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar_or_vector(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            // Get image info and validate Dim
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                // QuerySizeLod not valid for Rect, MS, or Buffer
                                if info.dim == Dim::DimRect
                                    || info.multisampled != 0
                                    || info.dim == Dim::DimBuffer
                                {
                                    return Err(ValidationError::ImageQuerySizeLodInvalidDim {
                                        function: function_id,
                                        block: block_id,
                                        dim: info.dim,
                                    }
                                    .into());
                                }
                            }
                        }

                        Op::ImageQuerySize => {
                            // Result must be int scalar or vector
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar_or_vector(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar or vector",
                                    }
                                    .into());
                                }
                            }

                            // Only valid for MS, Rect, or Buffer, or sampled=0/2
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                let valid = info.multisampled != 0
                                    || info.dim == Dim::DimRect
                                    || info.dim == Dim::DimBuffer
                                    || info.sampled != 1;

                                if !valid {
                                    return Err(ValidationError::ImageQuerySizeInvalidDim {
                                        function: function_id,
                                        block: block_id,
                                        dim: info.dim,
                                    }
                                    .into());
                                }
                            }
                        }

                        Op::ImageQueryLod => {
                            // Execution model check: must be Fragment, GLCompute, MeshEXT, or TaskEXT
                            let valid_models = [
                                ExecutionModel::Fragment,
                                ExecutionModel::GLCompute,
                                ExecutionModel::MeshEXT,
                                ExecutionModel::TaskEXT,
                            ];
                            let has_valid_model =
                                ctx.entry_models.iter().any(|m| valid_models.contains(m));
                            if !has_valid_model && !ctx.entry_models.is_empty() {
                                return Err(ValidationError::ImageQueryLodRequiresFragment {
                                    function: function_id,
                                    block: block_id,
                                }
                                .into());
                            }

                            // For GLCompute/MeshEXT/TaskEXT, require derivative group execution mode
                            let needs_derivative_mode = ctx.entry_models.iter().any(|m| {
                                matches!(
                                    m,
                                    ExecutionModel::GLCompute
                                        | ExecutionModel::MeshEXT
                                        | ExecutionModel::TaskEXT
                                )
                            });
                            if needs_derivative_mode
                                && !ctx.entry_models.contains(&ExecutionModel::Fragment)
                            {
                                let has_derivative_mode =
                                    ctx.module.execution_modes.iter().any(|mode_inst| {
                                        mode_inst.operands.get(1).is_some_and(|operand| {
                                            matches!(
                                                operand,
                                                Operand::ExecutionMode(
                                                    ExecutionMode::DerivativeGroupQuadsKHR
                                                ) | Operand::ExecutionMode(
                                                    ExecutionMode::DerivativeGroupLinearKHR
                                                )
                                            )
                                        })
                                    });
                                if !has_derivative_mode {
                                    return Err(ValidationError::ImageQueryLodRequiresFragment {
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into());
                                }
                            }

                            // Result must be float vector of 2 components
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions)
                                {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "float vector",
                                    }
                                    .into());
                                }
                                // Validate vector component count is 2
                                // Look up the type instruction to get vector component count
                                if let Ok(type_result_id) = ResultId::try_from(result_type) {
                                    if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                                        if let Some(size) = type_inst.vector_component_count() {
                                            if size != 2 {
                                                return Err(ValidationError::ImageQueryLodResultSizeInvalid {
                                                    function: function_id,
                                                    block: block_id,
                                                    expected: 2,
                                                    actual: size,
                                                }.into());
                                            }
                                        }
                                    }
                                }
                            }

                            // OpImageQueryLod cannot be used with multisampled images
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                if info.multisampled != 0 {
                                    return Err(
                                        ValidationError::ImageQueryLodCannotUseMultisampled {
                                            function: function_id,
                                            block: block_id,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }

                        Op::ImageQueryLevels => {
                            // Result must be int scalar
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar",
                                    }
                                    .into());
                                }
                            }

                            // OpImageQueryLevels cannot be used with Buffer, Rect, or SubpassData
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                if info.dim == Dim::DimBuffer
                                    || info.dim == Dim::DimRect
                                    || info.dim == Dim::DimSubpassData
                                {
                                    return Err(ValidationError::ImageQueryLevelsInvalidDim {
                                        function: function_id,
                                        block: block_id,
                                        dim: info.dim,
                                    }
                                    .into());
                                }
                            }
                        }

                        Op::ImageQuerySamples => {
                            // Result must be int scalar
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar",
                                    }
                                    .into());
                                }
                            }

                            // OpImageQuerySamples requires multisampled image
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                if info.multisampled == 0 {
                                    return Err(
                                        ValidationError::ImageQuerySamplesRequiresMultisampled {
                                            function: function_id,
                                            block: block_id,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }

                        Op::ImageQueryFormat | Op::ImageQueryOrder => {
                            // Result must be int scalar
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar",
                                    }
                                    .into());
                                }
                            }

                            // Operand must be OpTypeImage (not sampled image)
                            if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                                // Dim cannot be TileImageDataEXT
                                if info.dim == Dim::DimTileImageDataEXT {
                                    return Err(
                                        ValidationError::ImageQueryFormatOrderTileImageDataEXT {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into(),
                                    );
                                }
                            } else {
                                // Could not extract image type - operand is not an image
                                return Err(ValidationError::ImageQueryFormatOrderNotImage {
                                    function: function_id,
                                    block: block_id,
                                    opcode,
                                }
                                .into());
                            }
                        }

                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Sampled Image Rule
// ============================================================================

/// Validates OpSampledImage instructions.
pub struct SampledImageRule;

impl ValidationRule for SampledImageRule {
    fn name(&self) -> &'static str {
        "sampled-image"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode != Op::SampledImage {
                        continue;
                    }

                    // Validate result type is OpTypeSampledImage
                    if let Some(result_type) = inst.result_type {
                        if let Ok(type_id) = ResultId::try_from(result_type) {
                            if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                if type_inst.class.opcode != Op::TypeSampledImage {
                                    return Err(
                                        ValidationError::SampledImageResultTypeMustBeSampledImage {
                                            function: function_id,
                                            block: block_id,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }

                    // Get image type info and validate Sampled flag
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        // In Vulkan, Sampled must be 1
                        if ctx.is_vulkan_env() && info.sampled != 1 {
                            return Err(ValidationError::SampledImageRequiresSampledOne {
                                function: function_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // SubpassData dimension cannot be used with OpSampledImage
                        if info.dim == Dim::DimSubpassData {
                            return Err(ValidationError::SampledImageCannotUseSubpassData {
                                function: function_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // In SPIR-V 1.6+, Buffer dimension is not allowed
                        if (ctx.target_version.major() > 1
                            || (ctx.target_version.major() == 1 && ctx.target_version.minor() >= 6))
                            && info.dim == Dim::DimBuffer
                        {
                            return Err(ValidationError::SampledImageBufferDimInvalid {
                                function: function_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    // Validate image operand type matches result type (except depth)
                    if let (Some(result_type), Some(Operand::IdRef(image_id))) =
                        (inst.result_type, inst.operands.first())
                    {
                        if let Ok(result_type_id) = ResultId::try_from(result_type) {
                            if let Some(result_type_inst) = ctx.definitions.get(&result_type_id) {
                                // Get the image type from the result (OpTypeSampledImage)
                                if let Some(Operand::IdRef(expected_image_type)) =
                                    result_type_inst.operands.first()
                                {
                                    // Get the actual image type from the operand
                                    if let Ok(image_result_id) = ResultId::try_from(*image_id) {
                                        if let Some(image_inst) =
                                            ctx.definitions.get(&image_result_id)
                                        {
                                            if let Some(actual_image_type) = image_inst.result_type
                                            {
                                                // Image types should match (except depth is allowed to differ)
                                                if *expected_image_type != actual_image_type {
                                                    // Check if they only differ in depth
                                                    if let (
                                                        Some(expected_info),
                                                        Some(actual_info),
                                                    ) = (
                                                        ImageTypeInfo::from_type_id(
                                                            *expected_image_type,
                                                            ctx,
                                                        ),
                                                        ImageTypeInfo::from_type_id(
                                                            actual_image_type,
                                                            ctx,
                                                        ),
                                                    ) {
                                                        // All fields except depth must match
                                                        if expected_info.sampled_type
                                                            != actual_info.sampled_type
                                                            || expected_info.dim != actual_info.dim
                                                            || expected_info.arrayed
                                                                != actual_info.arrayed
                                                            || expected_info.multisampled
                                                                != actual_info.multisampled
                                                            || expected_info.sampled
                                                                != actual_info.sampled
                                                            || expected_info.format
                                                                != actual_info.format
                                                        {
                                                            return Err(
                                                                ValidationError::SampledImageOperandTypeMismatch {
                                                                    function: function_id,
                                                                    block: block_id,
                                                                }.into(),
                        );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Validate Sampler operand is OpTypeSampler
                    if let Some(Operand::IdRef(sampler_id)) = inst.operands.get(1) {
                        if let Ok(sampler_result_id) = ResultId::try_from(*sampler_id) {
                            if let Some(sampler_inst) = ctx.definitions.get(&sampler_result_id) {
                                if let Some(sampler_type) = sampler_inst.result_type {
                                    if let Ok(sampler_type_id) = ResultId::try_from(sampler_type) {
                                        if let Some(sampler_type_inst) =
                                            ctx.definitions.get(&sampler_type_id)
                                        {
                                            if sampler_type_inst.class.opcode != Op::TypeSampler {
                                                return Err(ValidationError::SampledImageSamplerMustBeSamplerType {
                                                    function: function_id,
                                                    block: block_id,
                                                }.into());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Texel Pointer Rule
// ============================================================================

/// Validates OpImageTexelPointer instructions.
pub struct ImageTexelPointerRule;

impl ValidationRule for ImageTexelPointerRule {
    fn name(&self) -> &'static str {
        "image-texel-pointer"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ImageTexelPointer {
                        continue;
                    }

                    // Validate result type is a pointer
                    if let Some(result_type) = inst.result_type {
                        if let Ok(type_id) = ResultId::try_from(result_type) {
                            if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                if type_inst.class.opcode != Op::TypePointer
                                    && type_inst.class.opcode != Op::TypeUntypedPointerKHR
                                {
                                    return Err(
                                        ValidationError::ImageTexelPointerResultMustBePointer {
                                            function: function_id,
                                            block: block_id,
                                        }
                                        .into(),
                                    );
                                }

                                // Validate storage class is Image
                                if let Some(Operand::StorageClass(sc)) = type_inst.operands.first()
                                {
                                    if *sc != rspirv::spirv::StorageClass::Image {
                                        return Err(ValidationError::ImageTexelPointerStorageClassMustBeImage {
                                            function: function_id,
                                            block: block_id,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }

                    // Validate Coordinate is integer scalar or vector
                    if let Some(Operand::IdRef(coord_id)) = inst.operands.get(1) {
                        if let Ok(coord_result_id) = ResultId::try_from(*coord_id) {
                            if let Some(coord_inst) = ctx.definitions.get(&coord_result_id) {
                                if let Some(coord_type) = coord_inst.result_type {
                                    if !resolver
                                        .is_int_scalar_or_vector(coord_type, ctx.definitions)
                                    {
                                        return Err(ValidationError::ImageTexelPointerCoordMustBeIntScalarOrVector {
                                            function: function_id,
                                            block: block_id,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }

                    // Validate Sample is integer scalar
                    if let Some(Operand::IdRef(sample_id)) = inst.operands.get(2) {
                        if let Ok(sample_result_id) = ResultId::try_from(*sample_id) {
                            if let Some(sample_inst) = ctx.definitions.get(&sample_result_id) {
                                if let Some(sample_type) = sample_inst.result_type {
                                    if !resolver.is_int_scalar(sample_type, ctx.definitions) {
                                        return Err(ValidationError::ImageTexelPointerSampleMustBeIntScalar {
                                            function: function_id,
                                            block: block_id,
                                        }.into());
                                    }
                                }
                            }
                        }
                    }

                    // Get image type info and validate dimensions
                    if let Some(info) = get_image_type_from_texel_pointer(inst, ctx) {
                        // SubpassData cannot be used with ImageTexelPointer
                        if info.dim == Dim::DimSubpassData {
                            return Err(ValidationError::ImageTexelPointerCannotUseSubpassData {
                                function: function_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // TileImageDataEXT cannot be used with ImageTexelPointer
                        if info.dim == Dim::DimTileImageDataEXT {
                            return Err(ValidationError::ImageTexelPointerCannotUseTileImageData {
                                function: function_id,
                                block: block_id,
                            }
                            .into());
                        }

                        // For non-multisampled images (MS=0), Sample must be constant 0
                        if info.multisampled == 0 {
                            // Get the Sample operand (operand index 2)
                            if let Some(Operand::IdRef(sample_id)) = inst.operands.get(2) {
                                // Check if it's a constant with value 0
                                if let Ok(sample_result_id) = ResultId::try_from(*sample_id) {
                                    if let Some(sample_inst) =
                                        ctx.definitions.get(&sample_result_id)
                                    {
                                        let is_constant_zero = matches!(
                                            sample_inst.class.opcode,
                                            Op::Constant | Op::ConstantNull
                                        ) && sample_inst
                                            .operands
                                            .first()
                                            .map(|op| match op {
                                                Operand::LiteralBit32(v) => *v == 0,
                                                Operand::LiteralBit64(v) => *v == 0,
                                                _ => false,
                                            })
                                            .unwrap_or(
                                                sample_inst.class.opcode == Op::ConstantNull,
                                            );

                                        if !is_constant_zero {
                                            return Err(ValidationError::ImageTexelPointerSampleMustBeZeroForNonMultisampled {
                                                function: function_id,
                                                block: block_id,
                                            }.into());
                                        }
                                    }
                                }
                            }
                        }

                        // Validate format for Vulkan
                        if ctx.is_vulkan_env() {
                            let valid_format = matches!(
                                info.format,
                                ImageFormat::R64i
                                    | ImageFormat::R64ui
                                    | ImageFormat::R32f
                                    | ImageFormat::R32i
                                    | ImageFormat::R32ui
                            );
                            if !valid_format {
                                return Err(
                                    ValidationError::ImageTexelPointerFormatInvalidForVulkan {
                                        function: function_id,
                                        block: block_id,
                                        format: info.format,
                                    }
                                    .into(),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Type Sampled Image Rule
// ============================================================================

/// Validates OpTypeSampledImage instructions.
pub struct TypeSampledImageRule;

impl ValidationRule for TypeSampledImageRule {
    fn name(&self) -> &'static str {
        "type-sampled-image"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeSampledImage {
                continue;
            }

            let type_id = inst.result_id.and_then(|id| TypeId::try_from(id).ok());

            // Validate that operand is OpTypeImage
            let image_type_id = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            let image_result_id = match ResultId::try_from(image_type_id) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let image_type_inst = match ctx.definitions.get(&image_result_id) {
                Some(inst) => inst,
                None => continue,
            };

            if image_type_inst.class.opcode != Op::TypeImage {
                return Err(ValidationError::TypeSampledImageOperandMustBeImage { type_id }.into());
            }

            // Get image type info
            if let Some(info) = ImageTypeInfo::from_type_id(image_type_id, ctx) {
                // Sampled must be 0 or 1
                if info.sampled != 0 && info.sampled != 1 {
                    return Err(ValidationError::TypeSampledImageSampledMustBeZeroOrOne {
                        type_id,
                    }
                    .into());
                }

                // In SPIR-V 1.6+, Buffer dimension is not allowed
                if (ctx.target_version.major() > 1
                    || (ctx.target_version.major() == 1 && ctx.target_version.minor() >= 6))
                    && info.dim == Dim::DimBuffer
                {
                    return Err(
                        ValidationError::TypeSampledImageBufferDimInvalid { type_id }.into(),
                    );
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Rule (OpImage)
// ============================================================================

/// Validates OpImage instructions.
pub struct ImageRule;

impl ValidationRule for ImageRule {
    fn name(&self) -> &'static str {
        "image"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode != Op::Image {
                        continue;
                    }

                    // Validate result type is OpTypeImage
                    if let Some(result_type) = inst.result_type {
                        if let Ok(type_id) = ResultId::try_from(result_type) {
                            if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                if type_inst.class.opcode != Op::TypeImage {
                                    return Err(ValidationError::ImageResultTypeMustBeImage {
                                        function: function_id,
                                        block: block_id,
                                    }
                                    .into());
                                }
                            }
                        }
                    }

                    // Validate operand is OpTypeSampledImage
                    if let Some(Operand::IdRef(sampled_image_id)) = inst.operands.first() {
                        if let Ok(sampled_image_result_id) = ResultId::try_from(*sampled_image_id) {
                            if let Some(sampled_image_inst) =
                                ctx.definitions.get(&sampled_image_result_id)
                            {
                                if let Some(operand_type) = sampled_image_inst.result_type {
                                    if let Ok(operand_type_id) = ResultId::try_from(operand_type) {
                                        if let Some(operand_type_inst) =
                                            ctx.definitions.get(&operand_type_id)
                                        {
                                            if operand_type_inst.class.opcode
                                                != Op::TypeSampledImage
                                            {
                                                return Err(
                                                    ValidationError::ImageOperandMustBeSampledImage {
                                                        function: function_id,
                                                        block: block_id,
                                                    }.into(),
                        );
                                            }

                                            // Validate inner image type matches result type
                                            if let Some(Operand::IdRef(inner_image_type)) =
                                                operand_type_inst.operands.first()
                                            {
                                                if let Some(result_type) = inst.result_type {
                                                    if *inner_image_type != result_type {
                                                        return Err(
                                                            ValidationError::ImageSampledImageTypeMismatch {
                                                                function: function_id,
                                                                block: block_id,
                                                            }.into(),
                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Sparse Texels Resident Rule
// ============================================================================

/// Validates OpImageSparseTexelsResident instructions.
pub struct ImageSparseTexelsResidentRule;

impl ValidationRule for ImageSparseTexelsResidentRule {
    fn name(&self) -> &'static str {
        "image-sparse-texels-resident"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ImageSparseTexelsResident {
                        continue;
                    }

                    // Validate result type is bool scalar
                    if let Some(result_type) = inst.result_type {
                        if !resolver.is_bool_scalar(result_type, ctx.definitions) {
                            return Err(
                                ValidationError::ImageSparseTexelsResidentResultMustBeBool {
                                    function: function_id,
                                    block: block_id,
                                }
                                .into(),
                            );
                        }
                    }

                    // Validate Resident Code is int scalar
                    if let Some(Operand::IdRef(code_id)) = inst.operands.first() {
                        if let Ok(code_result_id) = ResultId::try_from(*code_id) {
                            if let Some(code_inst) = ctx.definitions.get(&code_result_id) {
                                if let Some(code_type) = code_inst.result_type {
                                    if !resolver.is_int_scalar(code_type, ctx.definitions) {
                                        return Err(
                                            ValidationError::ImageSparseTexelsResidentCodeMustBeInt {
                                                function: function_id,
                                                block: block_id,
                                            }.into(),
                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Sparse Image Sample Result Type Rule
// ============================================================================

/// Sparse image sample opcodes that require struct result type.
const SPARSE_IMAGE_SAMPLE_OPCODES: &[Op] = &[
    Op::ImageSparseSampleImplicitLod,
    Op::ImageSparseSampleExplicitLod,
    Op::ImageSparseSampleDrefImplicitLod,
    Op::ImageSparseSampleDrefExplicitLod,
    Op::ImageSparseFetch,
    Op::ImageSparseGather,
    Op::ImageSparseDrefGather,
    Op::ImageSparseRead,
];

/// Validates that sparse image sample operations have valid result types.
///
/// Sparse image operations must have a result type that is a struct with exactly
/// two members:
/// - Member 0: Residency code (must be int scalar)
/// - Member 1: Texel value (type depends on the image format)
pub struct SparseSampleResultTypeRule;

impl ValidationRule for SparseSampleResultTypeRule {
    fn name(&self) -> &'static str {
        "sparse-sample-result-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !SPARSE_IMAGE_SAMPLE_OPCODES.contains(&inst.class.opcode) {
                        continue;
                    }

                    let opcode = inst.class.opcode;

                    // Get the result type
                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result type must be a struct
                    let Ok(result_type_result_id) = ResultId::try_from(result_type_id) else {
                        continue;
                    };
                    let Some(type_inst) = ctx.definitions.get(&result_type_result_id) else {
                        continue;
                    };

                    if type_inst.class.opcode != Op::TypeStruct {
                        return Err(ValidationError::ImageSparseSampleResultMustBeStruct {
                            function: function_id,
                            block: block_id,
                            opcode,
                        }
                        .into());
                    }

                    // Struct must have exactly 2 members
                    let member_count = type_inst.operands.len();
                    if member_count != 2 {
                        return Err(ValidationError::ImageSparseSampleResultMustBeStruct {
                            function: function_id,
                            block: block_id,
                            opcode,
                        }
                        .into());
                    }

                    // First member (residency code) must be int scalar
                    if let Some(Operand::IdRef(member_type_id)) = type_inst.operands.first() {
                        if !resolver.is_int_scalar(*member_type_id, ctx.definitions) {
                            return Err(ValidationError::ImageSparseSampleResidencyMustBeInt {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get image type info from an image instruction.
fn get_image_type_from_instruction(
    inst: &Instruction,
    ctx: &ValidationContext<'_>,
) -> Option<ImageTypeInfo> {
    // The image/sampled-image operand is typically the first operand for most image ops
    let image_operand = inst.operands.first()?;
    let image_id = image_operand.id_ref_any()?;

    // Look up the instruction that defines this ID
    let image_result_id = ResultId::try_from(image_id).ok()?;
    let image_inst = ctx.definitions.get(&image_result_id)?;

    // Get the type of this operand
    let type_id = image_inst.result_type?;

    ImageTypeInfo::from_type_id(type_id, ctx)
}

/// Find the image operand mask in an instruction.
/// Returns the mask and the index where operand parameters start.
fn find_image_operand_mask(inst: &Instruction) -> Option<(ImageOperands, usize)> {
    // Image operand mask location varies by opcode
    // For most ops: after coordinate operand
    // The mask is identified by being an ImageOperands operand

    for (idx, operand) in inst.operands.iter().enumerate() {
        if let Operand::ImageOperands(mask) = operand {
            return Some((*mask, idx + 1));
        }
    }

    // No explicit mask means empty mask
    Some((ImageOperands::empty(), inst.operands.len()))
}

/// Get image type info for OpImageTexelPointer.
/// The image is accessed through a pointer operand (operand 0).
fn get_image_type_from_texel_pointer(
    inst: &Instruction,
    ctx: &ValidationContext<'_>,
) -> Option<ImageTypeInfo> {
    // OpImageTexelPointer: operand 0 is Image (a pointer to an image)
    let image_ptr_id = inst.operands.first()?.id_ref_any()?;
    let image_ptr_result_id = ResultId::try_from(image_ptr_id).ok()?;
    let image_ptr_inst = ctx.definitions.get(&image_ptr_result_id)?;

    // Get the type of the pointer (should be OpTypePointer)
    let ptr_type_id = image_ptr_inst.result_type?;
    let ptr_type_result_id = ResultId::try_from(ptr_type_id).ok()?;
    let ptr_type_inst = ctx.definitions.get(&ptr_type_result_id)?;

    if ptr_type_inst.class.opcode != Op::TypePointer {
        return None;
    }

    // Get the pointed-to type (should be OpTypeImage)
    let image_type_id = ptr_type_inst.operands.get(1)?.id_ref_any()?;

    ImageTypeInfo::from_type_id(image_type_id, ctx)
}

// ============================================================================
// Reserved/Invalid Image Opcodes Rule
// ============================================================================

/// Reserved sparse projection sampling opcodes that should never be used.
const RESERVED_IMAGE_OPCODES: &[Op] = &[
    Op::ImageSparseSampleProjImplicitLod,
    Op::ImageSparseSampleProjExplicitLod,
    Op::ImageSparseSampleProjDrefImplicitLod,
    Op::ImageSparseSampleProjDrefExplicitLod,
];

/// Validates that reserved image opcodes are not used.
///
/// These instructions are enabled by a capability but are reserved and
/// should never actually be used in valid SPIR-V modules.
pub struct ReservedImageOpcodeRule;

impl ValidationRule for ReservedImageOpcodeRule {
    fn name(&self) -> &'static str {
        "reserved-image-opcode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if RESERVED_IMAGE_OPCODES.contains(&inst.class.opcode) {
                        return Err(ValidationError::ReservedOpcodeUsed {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
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
// QCOM Image Processing Rule
// ============================================================================

/// Validates QCOM image processing instructions (SPV_QCOM_image_processing).
///
/// These instructions require specific decorations on their texture/sampler operands:
/// - OpImageSampleWeightedQCOM: weight operand needs WeightTextureQCOM
/// - OpImageBlockMatchSSDQCOM/SADQCOM: target and reference need BlockMatchTextureQCOM
/// - OpImageBlockMatchWindowSSDQCOM/SADQCOM: similar requirements
/// - OpImageBlockMatchGatherSSDQCOM/SADQCOM: similar requirements
pub struct QCOMImageProcessingRule;

impl QCOMImageProcessingRule {
    /// Checks if the given ID has the required decoration on its underlying variable.
    fn check_decoration(
        &self,
        ctx: &ValidationContext<'_>,
        operand_id: u32,
        required_decoration: Decoration,
        function_id: Option<Id>,
        block_id: Option<Id>,
        opcode: Op,
    ) -> ValidationResult {
        let operand_result_id = match ResultId::try_from(operand_id) {
            Ok(id) => id,
            Err(_) => return Ok(()), // Skip if invalid ID
        };

        let operand_inst = match ctx.definitions.get(&operand_result_id) {
            Some(inst) => inst,
            None => return Ok(()), // Skip if instruction not found
        };

        // Check if this is OpSampledImage - if so, we need to check both texture and sampler
        if operand_inst.class.opcode == Op::SampledImage {
            // Check the image operand (texture)
            if let Some(Operand::IdRef(texture_id)) = operand_inst.operands.first() {
                self.check_load_decoration(
                    ctx,
                    *texture_id,
                    Decoration::BlockMatchTextureQCOM,
                    function_id,
                    block_id,
                    opcode,
                )?;
            }
            // Check the sampler operand
            if let Some(Operand::IdRef(sampler_id)) = operand_inst.operands.get(1) {
                self.check_load_decoration(
                    ctx,
                    *sampler_id,
                    Decoration::BlockMatchSamplerQCOM,
                    function_id,
                    block_id,
                    opcode,
                )?;
            }
        } else {
            // Not a SampledImage, check the operand directly
            self.check_load_decoration(
                ctx,
                operand_id,
                required_decoration,
                function_id,
                block_id,
                opcode,
            )?;
        }

        Ok(())
    }

    /// Checks if the operand is an OpLoad and the underlying variable has the decoration.
    fn check_load_decoration(
        &self,
        ctx: &ValidationContext<'_>,
        operand_id: u32,
        required_decoration: Decoration,
        function_id: Option<Id>,
        block_id: Option<Id>,
        opcode: Op,
    ) -> ValidationResult {
        let operand_result_id = match ResultId::try_from(operand_id) {
            Ok(id) => id,
            Err(_) => return Ok(()),
        };

        let operand_inst = match ctx.definitions.get(&operand_result_id) {
            Some(inst) => inst,
            None => return Ok(()),
        };

        // Expect OpLoad
        if operand_inst.class.opcode != Op::Load {
            return Err(ValidationError::QCOMImageExpectsOpLoad {
                function: function_id,
                block: block_id,
                opcode,
            }
            .into());
        }

        // Get the variable being loaded
        let variable_id = match operand_inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => return Ok(()),
        };

        // Check if the variable has the required decoration
        if !self.has_decoration(ctx, variable_id, required_decoration) {
            return Err(ValidationError::QCOMImageMissingDecoration {
                function: function_id,
                block: block_id,
                opcode,
                decoration: required_decoration,
            }
            .into());
        }

        Ok(())
    }

    /// Checks if an ID has a specific decoration in the module's annotations.
    fn has_decoration(&self, ctx: &ValidationContext<'_>, id: u32, decoration: Decoration) -> bool {
        for inst in &ctx.module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let target = match inst.operands.first() {
                Some(Operand::IdRef(target_id)) => *target_id,
                _ => continue,
            };
            if target != id {
                continue;
            }
            let decor = match inst.operands.get(1) {
                Some(Operand::Decoration(d)) => *d,
                _ => continue,
            };
            if decor == decoration {
                return true;
            }
        }
        false
    }
}

impl ValidationRule for QCOMImageProcessingRule {
    fn name(&self) -> &'static str {
        "qcom-image-processing"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    match opcode {
                        Op::ImageSampleWeightedQCOM => {
                            // Weight operand (operand 4) needs WeightTextureQCOM
                            if let Some(Operand::IdRef(weight_id)) = inst.operands.get(4) {
                                self.check_decoration(
                                    ctx,
                                    *weight_id,
                                    Decoration::WeightTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                        }
                        Op::ImageBlockMatchSSDQCOM | Op::ImageBlockMatchSADQCOM => {
                            // Target (operand 2) and Reference (operand 4) need BlockMatchTextureQCOM
                            if let Some(Operand::IdRef(target_id)) = inst.operands.get(2) {
                                self.check_decoration(
                                    ctx,
                                    *target_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                            if let Some(Operand::IdRef(ref_id)) = inst.operands.get(4) {
                                self.check_decoration(
                                    ctx,
                                    *ref_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                        }
                        Op::ImageBlockMatchWindowSSDQCOM | Op::ImageBlockMatchWindowSADQCOM => {
                            // These also need validation but with potentially different decorations
                            if let Some(Operand::IdRef(target_id)) = inst.operands.get(2) {
                                self.check_decoration(
                                    ctx,
                                    *target_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                            if let Some(Operand::IdRef(ref_id)) = inst.operands.get(4) {
                                self.check_decoration(
                                    ctx,
                                    *ref_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                        }
                        Op::ImageBlockMatchGatherSSDQCOM | Op::ImageBlockMatchGatherSADQCOM => {
                            // Gather variants also need BlockMatchTextureQCOM
                            if let Some(Operand::IdRef(target_id)) = inst.operands.get(2) {
                                self.check_decoration(
                                    ctx,
                                    *target_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                            if let Some(Operand::IdRef(ref_id)) = inst.operands.get(4) {
                                self.check_decoration(
                                    ctx,
                                    *ref_id,
                                    Decoration::BlockMatchTextureQCOM,
                                    function_id,
                                    block_id,
                                    opcode,
                                )?;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Sampled Image Consumer Rule
// ============================================================================

/// Validates that OpSampledImage results are consumed in the same block.
///
/// SPIR-V requires:
/// 1. All OpSampledImage instructions must be in the same block in which their
///    Result <id> are consumed.
/// 2. Result <id> from OpSampledImage instructions must not appear as operands
///    to OpPhi instructions or OpSelect instructions.
pub struct SampledImageConsumerRule;

impl ValidationRule for SampledImageConsumerRule {
    fn name(&self) -> &'static str {
        "sampled-image-consumer"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            // Build a map of OpSampledImage result IDs to their block IDs
            let mut sampled_image_blocks: HashMap<u32, Option<Id>> = HashMap::new();

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if inst.class.opcode == Op::SampledImage {
                        if let Some(result_id) = inst.result_id {
                            sampled_image_blocks.insert(result_id, block_id);
                        }
                    }
                }
            }

            // Now check all consumers of OpSampledImage results
            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Check all IdRef operands
                    for operand in &inst.operands {
                        if let Operand::IdRef(id) = operand {
                            // Check if this is a reference to an OpSampledImage result
                            if let Some(def_block) = sampled_image_blocks.get(id) {
                                let sampled_image_id = Id::try_from(*id)
                                    .unwrap_or_else(|_| Id::try_from(0u32).unwrap());

                                // Check 1: OpPhi and OpSelect cannot use OpSampledImage results
                                if opcode == Op::Phi || opcode == Op::Select {
                                    return Err(ValidationError::SampledImageUsedInPhiOrSelect {
                                        function: function_id,
                                        block: block_id,
                                        sampled_image_id,
                                        consumer_opcode: opcode,
                                    }
                                    .into());
                                }

                                // Check 2: Consumer must be in the same block as the definition
                                if *def_block != block_id {
                                    return Err(
                                        ValidationError::SampledImageConsumedInDifferentBlock {
                                            function: function_id,
                                            def_block: *def_block,
                                            consumer_block: block_id,
                                            sampled_image_id,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Dref Rule
// ============================================================================

/// Validates image Dref (depth reference) operations.
///
/// SPIR-V and Vulkan requirements for Dref operations:
/// 1. The Dref operand must be a 32-bit float scalar
/// 2. In Vulkan, Dref operations cannot use images with 3D dimension
/// 3. Dref operations cannot use multisampled images
pub struct ImageDrefRule;

impl ValidationRule for ImageDrefRule {
    fn name(&self) -> &'static str {
        "image-dref"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Check if this is a Dref operation
                    if !opcode.is_dref() {
                        continue;
                    }

                    // Get image type info to check dimension and multisampling
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        // Check: Vulkan forbids 3D dimension for Dref operations
                        if ctx.is_vulkan_env() && info.dim == Dim::Dim3D {
                            return Err(ValidationError::ImageDrefCannotUse3DInVulkan {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }

                        // Check: Dref operations cannot use multisampled images
                        if info.multisampled != 0 {
                            return Err(ValidationError::ImageDrefCannotUseMultisample {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Get the Dref operand type and validate it's a 32-bit float scalar
                    // For Dref operations, the Dref operand is at index 2 (after Sampled Image and Coordinate)
                    if let Some(Operand::IdRef(dref_id)) = inst.operands.get(2) {
                        if let Ok(dref_result_id) = ResultId::try_from(*dref_id) {
                            if let Some(dref_inst) = ctx.definitions.get(&dref_result_id) {
                                if let Some(dref_type_id) = dref_inst.result_type {
                                    // Check if it's a 32-bit float scalar
                                    let is_float_scalar =
                                        resolver.is_float_scalar(dref_type_id, ctx.definitions);
                                    let bit_width =
                                        resolver.get_bit_width(dref_type_id, ctx.definitions);

                                    if !is_float_scalar || bit_width != Some(32) {
                                        return Err(ValidationError::ImageDrefMustBe32BitFloat {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Proj Rule
// ============================================================================

/// Validates image Proj (projection) operations.
///
/// SPIR-V requirements for Proj operations:
/// 1. Image dimension must be 1D, 2D, 3D, or Rect (not Cube, Buffer, SubpassData)
/// 2. Cannot use multisampled images (MS must be 0)
/// 3. Cannot use arrayed images (Arrayed must be 0)
pub struct ImageProjRule;

impl ValidationRule for ImageProjRule {
    fn name(&self) -> &'static str {
        "image-proj"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Check if this is a Proj operation
                    if !opcode.is_proj() {
                        continue;
                    }

                    // Get image type info
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        // Check: Proj requires Dim 1D, 2D, 3D, or Rect
                        let valid_dim = matches!(
                            info.dim,
                            Dim::Dim1D | Dim::Dim2D | Dim::Dim3D | Dim::DimRect
                        );
                        if !valid_dim {
                            return Err(ValidationError::ImageProjInvalidDim {
                                function: function_id,
                                block: block_id,
                                opcode,
                                dim: info.dim,
                            }
                            .into());
                        }

                        // Check: Proj cannot use multisampled images
                        if info.multisampled != 0 {
                            return Err(ValidationError::ImageProjCannotUseMultisample {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }

                        // Check: Proj cannot use arrayed images
                        if info.arrayed != 0 {
                            return Err(ValidationError::ImageProjCannotUseArrayed {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Read Vulkan 4-Component Rule
// ============================================================================

/// Validates that OpImageRead result is 4-component vector in Vulkan.
///
/// Vulkan requirement (VUID-StandaloneSpirv-OpImageRead-04780):
/// In Vulkan, OpImageRead and OpImageSparseRead result type must be a
/// 4-component int or float vector.
pub struct ImageReadVulkan4ComponentRule;

/// Helper to get vector component count from a type ID.
fn get_vector_component_count_local(type_id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&result_id)?;
    if type_inst.class.opcode == Op::TypeVector {
        if let Some(Operand::LiteralBit32(count)) = type_inst.operands.get(1) {
            return Some(*count);
        }
    }
    // Scalar types have 1 component
    if matches!(type_inst.class.opcode, Op::TypeFloat | Op::TypeInt) {
        return Some(1);
    }
    None
}

impl ValidationRule for ImageReadVulkan4ComponentRule {
    fn name(&self) -> &'static str {
        "image-read-vulkan-4-component"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only applies to Vulkan environment
        if !ctx.is_vulkan_env() {
            return Ok(());
        }

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Check OpImageRead and OpImageSparseRead
                    if opcode != Op::ImageRead && opcode != Op::ImageSparseRead {
                        continue;
                    }

                    // Get result type
                    if let Some(result_type) = inst.result_type {
                        // For sparse operations, get the actual result type from struct
                        let actual_result_type = if opcode == Op::ImageSparseRead {
                            // Sparse result is a struct, second member is the actual result
                            if let Ok(result_id) = ResultId::try_from(result_type) {
                                if let Some(type_inst) = ctx.definitions.get(&result_id) {
                                    if type_inst.class.opcode == Op::TypeStruct {
                                        // Get the second member type (index 1)
                                        type_inst
                                            .operands
                                            .get(1)
                                            .and_then(|op| op.id_ref_any())
                                            .unwrap_or(result_type)
                                    } else {
                                        result_type
                                    }
                                } else {
                                    result_type
                                }
                            } else {
                                result_type
                            }
                        } else {
                            result_type
                        };

                        // Check component count
                        if let Some(component_count) =
                            get_vector_component_count_local(actual_result_type, ctx)
                        {
                            if component_count != 4 {
                                return Err(
                                    ValidationError::ImageReadResultMustBe4ComponentInVulkan {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        actual_components: component_count,
                                    }
                                    .into(),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Gather Validation Rule
// ============================================================================

/// Validates image gather operations.
///
/// SPIR-V and Vulkan requirements for gather operations:
/// 1. Image Dim must be 2D, Cube, or Rect (not 1D, 3D, Buffer, SubpassData)
/// 2. Component operand (for non-Dref gather) must be 32-bit int scalar
/// 3. In Vulkan, Component operand must be a constant (not a runtime value)
pub struct ImageGatherRule;

/// Check if an instruction is a constant opcode.
fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::ConstantNull
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::SpecConstant
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
    )
}

impl ValidationRule for ImageGatherRule {
    fn name(&self) -> &'static str {
        "image-gather"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Check if this is a gather operation
                    if !opcode.is_gather() {
                        continue;
                    }

                    // Validate dimension (applies to all gather operations)
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        let valid_dim =
                            matches!(info.dim, Dim::Dim2D | Dim::DimCube | Dim::DimRect);
                        if !valid_dim {
                            return Err(ValidationError::ImageGatherInvalidDim {
                                function: function_id,
                                block: block_id,
                                opcode,
                                dim: info.dim,
                            }
                            .into());
                        }
                    }

                    // For non-Dref gather (OpImageGather, OpImageSparseGather),
                    // validate Component operand (operand index 3)
                    if opcode == Op::ImageGather || opcode == Op::ImageSparseGather {
                        // Component is at operand index 3 (after Sampled Image, Coordinate, Component)
                        // Actually for OpImageGather: Sampled Image (0), Coordinate (1), Component (2)
                        if let Some(Operand::IdRef(component_id)) = inst.operands.get(2) {
                            if let Ok(component_result_id) = ResultId::try_from(*component_id) {
                                if let Some(component_inst) =
                                    ctx.definitions.get(&component_result_id)
                                {
                                    // Check Component is 32-bit int scalar
                                    if let Some(component_type_id) = component_inst.result_type {
                                        let is_int_scalar = resolver
                                            .is_int_scalar(component_type_id, ctx.definitions);
                                        let bit_width = resolver
                                            .get_bit_width(component_type_id, ctx.definitions);

                                        if !is_int_scalar || bit_width != Some(32) {
                                            return Err(
                                                ValidationError::ImageGatherComponentMustBe32BitInt {
                                                    function: function_id,
                                                    block: block_id,
                                                    opcode,
                                                }.into(),
                        );
                                        }
                                    }

                                    // Vulkan: Component must be constant
                                    if ctx.is_vulkan_env()
                                        && !is_constant_opcode(component_inst.class.opcode)
                                    {
                                        return Err(
                                                ValidationError::ImageGatherComponentMustBeConstantInVulkan {
                                                    function: function_id,
                                                    block: block_id,
                                                    opcode,
                                                }.into(),
                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Coordinate Rule
// ============================================================================

/// Returns true if the operation requires float-only coordinates.
fn requires_float_coords(op: Op) -> bool {
    op.is_implicit_lod() || op.is_gather() || op == Op::ImageQueryLod
}

/// Returns true if the operation requires int-only coordinates.
fn requires_int_coords(op: Op) -> bool {
    op.is_fetch() || op == Op::ImageTexelPointer
}

/// Returns true if the operation accepts either int or float coordinates.
fn accepts_int_or_float_coords(op: Op) -> bool {
    op.is_explicit_lod() || op.is_image_read_write()
}

/// Returns true if this opcode has a coordinate operand to validate.
fn has_coordinate_operand(op: Op) -> bool {
    requires_float_coords(op) || requires_int_coords(op) || accepts_int_or_float_coords(op)
}

/// Validates image coordinate types and component counts.
///
/// Checks:
/// - Coordinate type: float for sample/gather, int for fetch/texel pointer,
///   int or float for explicit LOD/read/write
/// - Coordinate width: must be 32-bit
/// - Component count: must have at least the required number of components
pub struct ImageCoordinateRule;

impl ValidationRule for ImageCoordinateRule {
    fn name(&self) -> &'static str {
        "image-coordinate"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !has_coordinate_operand(opcode) {
                        continue;
                    }

                    // Coordinate is at operand index 1 for most image ops
                    // (operand 0 = image/sampled-image, operand 1 = coordinate)
                    let coord_idx = 1;
                    let Some(Operand::IdRef(coord_id)) = inst.operands.get(coord_idx) else {
                        continue;
                    };
                    let Ok(coord_result_id) = ResultId::try_from(*coord_id) else {
                        continue;
                    };
                    let Some(coord_inst) = ctx.definitions.get(&coord_result_id) else {
                        continue;
                    };
                    let Some(coord_type_id) = coord_inst.result_type else {
                        continue;
                    };

                    // Type check: float vs int vs either
                    let is_float =
                        resolver.is_float_scalar_or_vector(coord_type_id, ctx.definitions);
                    let is_int = resolver.is_int_scalar_or_vector(coord_type_id, ctx.definitions);

                    if requires_float_coords(opcode) && !is_float {
                        return Err(ValidationError::ImageCoordinateTypeMismatch {
                            function: function_id,
                            block: block_id,
                            opcode,
                            expected: "float",
                        }
                        .into());
                    }
                    if requires_int_coords(opcode) && !is_int {
                        return Err(ValidationError::ImageCoordinateTypeMismatch {
                            function: function_id,
                            block: block_id,
                            opcode,
                            expected: "integer",
                        }
                        .into());
                    }
                    if accepts_int_or_float_coords(opcode) && !is_float && !is_int {
                        return Err(ValidationError::ImageCoordinateTypeMismatch {
                            function: function_id,
                            block: block_id,
                            opcode,
                            expected: "integer or float",
                        }
                        .into());
                    }

                    // Bit-width check: must be 32-bit
                    if let Some(bit_width) = resolver.get_bit_width(coord_type_id, ctx.definitions)
                    {
                        if bit_width != 32 {
                            return Err(ValidationError::ImageCoordinateNot32Bit {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Component count check
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        let required = get_min_coord_size(opcode, &info);
                        let actual = resolver.get_dimension(coord_type_id, ctx.definitions);

                        if actual < required {
                            return Err(ValidationError::ImageCoordinateInsufficientComponents {
                                function: function_id,
                                block: block_id,
                                opcode,
                                required,
                                actual,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Fetch Rule
// ============================================================================

/// Validates OpImageFetch and OpImageSparseFetch instructions.
///
/// Checks:
/// - Result type is a 4-component int/float vector
/// - Result component type matches image sampled type
/// - Image 'Sampled' parameter is 1 (sampling image, not storage)
/// - Image dimension is not Cube
pub struct ImageFetchRule;

impl ValidationRule for ImageFetchRule {
    fn name(&self) -> &'static str {
        "image-fetch"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    if !opcode.is_fetch() {
                        continue;
                    }

                    // Validate result type is a 4-component int/float vector
                    if let Some(result_type_id) = inst.result_type {
                        if let Ok(rt_id) = ResultId::try_from(result_type_id) {
                            if let Some(rt_inst) = ctx.definitions.get(&rt_id) {
                                if rt_inst.is_vector_type() {
                                    if rt_inst.vector_component_count() != Some(4) {
                                        return Err(
                                            ValidationError::ImageSampleResultMustBe4ComponentVector {
                                                function: function_id,
                                                block: block_id,
                                                opcode,
                                            }
                                            .into(),
                                        );
                                    }
                                } else {
                                    return Err(
                                        ValidationError::ImageSampleResultMustBe4ComponentVector {
                                            function: function_id,
                                            block: block_id,
                                            opcode,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }

                    // Get image type info and validate
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        // Image dimension cannot be Cube
                        if info.dim == Dim::DimCube {
                            return Err(ValidationError::ImageFetchDimCannotBeCube {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }

                        // Image 'Sampled' parameter must be 1
                        if info.sampled != 0 && info.sampled != 1 {
                            return Err(ValidationError::ImageFetchRequiresSampledImage {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }

                        // Result component type must match image sampled type
                        if info.sampled_type != 0 {
                            if let Some(result_type_id) = inst.result_type {
                                if let Ok(rt_id) = ResultId::try_from(result_type_id) {
                                    if let Some(rt_inst) = ctx.definitions.get(&rt_id) {
                                        if let Some(component_type_id) =
                                            rt_inst.vector_component_type_id()
                                        {
                                            if component_type_id != info.sampled_type {
                                                return Err(
                                                    ValidationError::ImageSampleResultTypeMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                        opcode,
                                                    }
                                                    .into(),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Image Sample Result Type Rule
// ============================================================================

/// Validates that image sample operations produce the correct result type.
///
/// For non-Dref sampling: result must be a 4-component vector of the sampled type.
/// For Dref sampling: result must be a scalar of the sampled type.
/// For gather operations: result must be a 4-component vector of the sampled type.
pub struct ImageSampleResultTypeRule;

impl ValidationRule for ImageSampleResultTypeRule {
    fn name(&self) -> &'static str {
        "image-sample-result-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let opcode = inst.class.opcode;

                    // Skip opcodes that aren't non-sparse sample/gather ops
                    // (fetch is handled by ImageFetchRule, sparse by SparseSampleResultTypeRule)
                    if !opcode.is_sample() && !opcode.is_gather() {
                        continue;
                    }
                    // Skip sparse variants (handled by SparseSampleResultTypeRule)
                    if opcode.is_sparse() {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };
                    let Ok(rt_id) = ResultId::try_from(result_type_id) else {
                        continue;
                    };
                    let Some(rt_inst) = ctx.definitions.get(&rt_id) else {
                        continue;
                    };

                    if opcode.is_dref() && !opcode.is_gather() {
                        // Dref (non-gather): result must be a scalar of sampled type
                        if rt_inst.is_vector_type() || rt_inst.is_matrix_type() {
                            return Err(ValidationError::ImageDrefSampleResultMustBeScalar {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    } else {
                        // Non-Dref sample / gather: result must be vec4
                        if rt_inst.is_vector_type() {
                            if rt_inst.vector_component_count() != Some(4) {
                                return Err(
                                    ValidationError::ImageSampleResultMustBe4ComponentVector {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                    }
                                    .into(),
                                );
                            }
                        } else {
                            return Err(ValidationError::ImageSampleResultMustBe4ComponentVector {
                                function: function_id,
                                block: block_id,
                                opcode,
                            }
                            .into());
                        }
                    }

                    // Validate component type matches image sampled type
                    if let Some(info) = get_image_type_from_instruction(inst, ctx) {
                        if info.sampled_type != 0 {
                            let actual_component_type = if opcode.is_dref() && !opcode.is_gather() {
                                // Scalar result: the result type ID itself is the component type
                                Some(result_type_id)
                            } else {
                                // Vector result: extract component type from vector
                                rt_inst.vector_component_type_id()
                            };

                            if let Some(component_type_id) = actual_component_type {
                                if component_type_id != info.sampled_type {
                                    return Err(ValidationError::ImageSampleResultTypeMismatch {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                    }
                                    .into());
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All Image Rules
// ============================================================================

/// Returns all image validation rules.
pub fn all_image_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ImageTypeRule,
        &TypeSampledImageRule,
        &ImageOperandRule,
        &ImageSampleExecutionModelRule,
        &ImageReadWriteRule,
        &ImageQueryRule,
        &SampledImageRule,
        &SampledImageConsumerRule,
        &ImageTexelPointerRule,
        &ImageRule,
        &ImageSparseTexelsResidentRule,
        &SparseSampleResultTypeRule,
        &ReservedImageOpcodeRule,
        &QCOMImageProcessingRule,
        &ImageDrefRule,
        &ImageProjRule,
        &ImageReadVulkan4ComponentRule,
        &ImageGatherRule,
        &ImageFetchRule,
        &ImageCoordinateRule,
        &ImageSampleResultTypeRule,
        &ImageOperandTypeRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_implicit_lod() {
        assert!(Op::ImageSampleImplicitLod.is_implicit_lod());
        assert!(Op::ImageSampleDrefImplicitLod.is_implicit_lod());
        assert!(!Op::ImageSampleExplicitLod.is_implicit_lod());
        assert!(!Op::ImageFetch.is_implicit_lod());
    }

    #[test]
    fn test_is_explicit_lod() {
        assert!(Op::ImageSampleExplicitLod.is_explicit_lod());
        assert!(Op::ImageSampleDrefExplicitLod.is_explicit_lod());
        assert!(!Op::ImageSampleImplicitLod.is_explicit_lod());
        assert!(!Op::ImageFetch.is_explicit_lod());
    }

    #[test]
    fn test_is_proj() {
        assert!(Op::ImageSampleProjImplicitLod.is_proj());
        assert!(Op::ImageSampleProjExplicitLod.is_proj());
        assert!(!Op::ImageSampleImplicitLod.is_proj());
        assert!(!Op::ImageFetch.is_proj());
    }

    #[test]
    fn test_is_gather() {
        assert!(Op::ImageGather.is_gather());
        assert!(Op::ImageDrefGather.is_gather());
        assert!(!Op::ImageFetch.is_gather());
        assert!(!Op::ImageSampleImplicitLod.is_gather());
    }

    #[test]
    fn test_get_plane_coord_size() {
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::Dim1D,
                ..Default::default()
            }),
            1
        );
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::DimBuffer,
                ..Default::default()
            }),
            1
        );
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::Dim2D,
                ..Default::default()
            }),
            2
        );
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::DimRect,
                ..Default::default()
            }),
            2
        );
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::Dim3D,
                ..Default::default()
            }),
            3
        );
        assert_eq!(
            get_plane_coord_size(&ImageTypeInfo {
                dim: Dim::DimCube,
                ..Default::default()
            }),
            3
        );
    }

    #[test]
    fn test_get_min_coord_size() {
        let info_2d = ImageTypeInfo {
            dim: Dim::Dim2D,
            arrayed: 0,
            ..Default::default()
        };
        let info_2d_array = ImageTypeInfo {
            dim: Dim::Dim2D,
            arrayed: 1,
            ..Default::default()
        };

        assert_eq!(get_min_coord_size(Op::ImageFetch, &info_2d), 2);
        assert_eq!(get_min_coord_size(Op::ImageFetch, &info_2d_array), 3);
        assert_eq!(
            get_min_coord_size(Op::ImageSampleProjImplicitLod, &info_2d),
            3
        );
    }
}
