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

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Capability, Dim, ExecutionModel, ImageFormat, ImageOperands, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::op_ext::OpExt;
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};

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
            sampled_type: image_inst.operands.get(0)?.id_ref_any()?,
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeImage {
                continue;
            }

            // Validate operand count
            if inst.operands.len() < 7 {
                return Err(ValidationError::ImageTypeInvalidOperandCount {
                    type_id: inst.result_id.map(|id| TypeId::try_from(id).ok()).flatten(),
                    expected: 7,
                    actual: inst.operands.len(),
                });
            }

            // Extract Dim
            let dim = match &inst.operands.get(1) {
                Some(Operand::Dim(d)) => *d,
                _ => continue,
            };

            // Extract sampled flag
            let sampled = match &inst.operands.get(5) {
                Some(Operand::LiteralBit32(v)) => *v,
                _ => continue,
            };

            // Extract format
            let format = match &inst.operands.get(6) {
                Some(Operand::ImageFormat(f)) => *f,
                _ => continue,
            };

            // Validate: SubpassData must have Dim = 2D, MS = 0 or 1, Sampled = 2, Arrayed = 0
            if dim == Dim::DimSubpassData {
                let arrayed = match &inst.operands.get(3) {
                    Some(Operand::LiteralBit32(v)) => *v,
                    _ => continue,
                };
                if arrayed != 0 {
                    return Err(ValidationError::ImageTypeSubpassDataMustNotBeArrayed {
                        type_id: inst.result_id.map(|id| TypeId::try_from(id).ok()).flatten(),
                    });
                }
                if sampled != 2 {
                    return Err(ValidationError::ImageTypeSubpassDataSampledMustBeTwo {
                        type_id: inst.result_id.map(|id| TypeId::try_from(id).ok()).flatten(),
                    });
                }
            }

            // Validate: Buffer images must have format != Unknown in Vulkan
            if dim == Dim::DimBuffer && format == ImageFormat::Unknown {
                if ctx.is_vulkan_env() {
                    return Err(ValidationError::ImageTypeBufferFormatRequired {
                        type_id: inst.result_id.map(|id| TypeId::try_from(id).ok()).flatten(),
                    });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {

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

                    // Validate multisampled images require Sample operand
                    if let Some(ref info) = image_type_info {
                        if info.multisampled != 0 && !mask.contains(ImageOperands::SAMPLE) {
                            return Err(ValidationError::ImageOperandSampleRequiredForMultisampled {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate mutually exclusive offset operands
                    let offset_count = [
                        mask.contains(ImageOperands::OFFSET),
                        mask.contains(ImageOperands::CONST_OFFSET),
                        mask.contains(ImageOperands::CONST_OFFSETS),
                    ].iter().filter(|&&x| x).count();

                    if offset_count > 1 {
                        return Err(ValidationError::ImageOperandMultipleOffsets {
                            function: function_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        });
                    }

                    // Validate Bias operand
                    if mask.contains(ImageOperands::BIAS) {
                        if !inst.class.opcode.is_implicit_lod() {
                            return Err(ValidationError::ImageOperandBiasRequiresImplicitLod {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate Lod operand
                    if mask.contains(ImageOperands::LOD) {
                        let valid_for_lod = inst.class.opcode.is_explicit_lod()
                            || inst.class.opcode.is_fetch()
                            || (inst.class.opcode.is_image_read_write()
                                && ctx.has_capability(Capability::ImageReadWriteLodAMD));

                        if !valid_for_lod {
                            return Err(ValidationError::ImageOperandLodRequiresExplicitLodOrFetch {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }

                        // Lod and Grad are mutually exclusive
                        if mask.contains(ImageOperands::GRAD) {
                            return Err(ValidationError::ImageOperandLodAndGradMutuallyExclusive {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate Grad operand
                    if mask.contains(ImageOperands::GRAD) {
                        if !inst.class.opcode.is_explicit_lod() {
                            return Err(ValidationError::ImageOperandGradRequiresExplicitLod {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate ConstOffsets operand
                    if mask.contains(ImageOperands::CONST_OFFSETS) {
                        if !inst.class.opcode.is_gather() {
                            return Err(ValidationError::ImageOperandConstOffsetsRequiresGather {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate offset operands cannot be used with Cube
                    if let Some(ref info) = image_type_info {
                        if info.dim == Dim::DimCube {
                            if mask.contains(ImageOperands::OFFSET)
                                || mask.contains(ImageOperands::CONST_OFFSET)
                                || mask.contains(ImageOperands::CONST_OFFSETS)
                            {
                                return Err(ValidationError::ImageOperandOffsetCannotBeUsedWithCube {
                                    function: function_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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

        // Implicit LOD is only valid in Fragment shader (or with derivative capability)
        let has_fragment = ctx.entry_models.contains(&ExecutionModel::Fragment);
        let has_derivative_capability = ctx.has_capability(Capability::DerivativeControl);

        if !has_fragment && !has_derivative_capability && !ctx.entry_models.is_empty() {
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
                            });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                });
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
                                });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                    });
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
                                    });
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
                                    });
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
                                    });
                                }
                            }
                        }

                        Op::ImageQueryLod => {
                            // Result must be float vector of 2 components
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "float vector",
                                    });
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
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Op::ImageQueryLevels | Op::ImageQuerySamples => {
                            // Result must be int scalar
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ImageQueryResultTypeInvalid {
                                        function: function_id,
                                        block: block_id,
                                        opcode,
                                        expected: "integer scalar",
                                    });
                                }
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                    return Err(ValidationError::SampledImageResultTypeMustBeSampledImage {
                                        function: function_id,
                                        block: block_id,
                                    });
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
                            });
                        }

                        // SubpassData dimension cannot be used with OpSampledImage
                        if info.dim == Dim::DimSubpassData {
                            return Err(ValidationError::SampledImageCannotUseSubpassData {
                                function: function_id,
                                block: block_id,
                            });
                        }

                        // In SPIR-V 1.6+, Buffer dimension is not allowed
                        if ctx.target_version.major() > 1
                            || (ctx.target_version.major() == 1
                                && ctx.target_version.minor() >= 6)
                        {
                            if info.dim == Dim::DimBuffer {
                                return Err(ValidationError::SampledImageBufferDimInvalid {
                                    function: function_id,
                                    block: block_id,
                                });
                            }
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
                                            if let Some(actual_image_type) = image_inst.result_type {
                                                // Image types should match (except depth is allowed to differ)
                                                if *expected_image_type != actual_image_type {
                                                    // Check if they only differ in depth
                                                    if let (Some(expected_info), Some(actual_info)) = (
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
                                                                },
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
                                                });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                    return Err(ValidationError::ImageTexelPointerResultMustBePointer {
                                        function: function_id,
                                        block: block_id,
                                    });
                                }

                                // Validate storage class is Image
                                if let Some(Operand::StorageClass(sc)) = type_inst.operands.first() {
                                    if *sc != rspirv::spirv::StorageClass::Image {
                                        return Err(ValidationError::ImageTexelPointerStorageClassMustBeImage {
                                            function: function_id,
                                            block: block_id,
                                        });
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
                                    if !resolver.is_int_scalar_or_vector(coord_type, ctx.definitions) {
                                        return Err(ValidationError::ImageTexelPointerCoordMustBeIntScalarOrVector {
                                            function: function_id,
                                            block: block_id,
                                        });
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
                                        });
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
                            });
                        }

                        // TileImageDataEXT cannot be used with ImageTexelPointer
                        if info.dim == Dim::DimTileImageDataEXT {
                            return Err(ValidationError::ImageTexelPointerCannotUseTileImageData {
                                function: function_id,
                                block: block_id,
                            });
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
                                return Err(ValidationError::ImageTexelPointerFormatInvalidForVulkan {
                                    function: function_id,
                                    block: block_id,
                                    format: info.format,
                                });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                return Err(ValidationError::TypeSampledImageOperandMustBeImage { type_id });
            }

            // Get image type info
            if let Some(info) = ImageTypeInfo::from_type_id(image_type_id, ctx) {
                // Sampled must be 0 or 1
                if info.sampled != 0 && info.sampled != 1 {
                    return Err(ValidationError::TypeSampledImageSampledMustBeZeroOrOne { type_id });
                }

                // In SPIR-V 1.6+, Buffer dimension is not allowed
                if ctx.target_version.major() > 1
                    || (ctx.target_version.major() == 1 && ctx.target_version.minor() >= 6)
                {
                    if info.dim == Dim::DimBuffer {
                        return Err(ValidationError::TypeSampledImageBufferDimInvalid { type_id });
                    }
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                    });
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
                                                    },
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
                                                            },
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                                },
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
                                            },
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        });
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
        &ImageTexelPointerRule,
        &ImageRule,
        &ImageSparseTexelsResidentRule,
        &ReservedImageOpcodeRule,
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
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::Dim1D, ..Default::default() }), 1);
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::DimBuffer, ..Default::default() }), 1);
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::Dim2D, ..Default::default() }), 2);
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::DimRect, ..Default::default() }), 2);
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::Dim3D, ..Default::default() }), 3);
        assert_eq!(get_plane_coord_size(&ImageTypeInfo { dim: Dim::DimCube, ..Default::default() }), 3);
    }

    #[test]
    fn test_get_min_coord_size() {
        let info_2d = ImageTypeInfo { dim: Dim::Dim2D, arrayed: 0, ..Default::default() };
        let info_2d_array = ImageTypeInfo { dim: Dim::Dim2D, arrayed: 1, ..Default::default() };

        assert_eq!(get_min_coord_size(Op::ImageFetch, &info_2d), 2);
        assert_eq!(get_min_coord_size(Op::ImageFetch, &info_2d_array), 3);
        assert_eq!(get_min_coord_size(Op::ImageSampleProjImplicitLod, &info_2d), 3);
    }
}
