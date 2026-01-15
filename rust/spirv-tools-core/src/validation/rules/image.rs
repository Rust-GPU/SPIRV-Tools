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

// ============================================================================
// Image Opcode Classification
// ============================================================================

/// Sample instructions that use implicit LOD.
const IMPLICIT_LOD_OPS: &[Op] = &[
    Op::ImageSampleImplicitLod,
    Op::ImageSampleDrefImplicitLod,
    Op::ImageSampleProjImplicitLod,
    Op::ImageSampleProjDrefImplicitLod,
    Op::ImageSparseSampleImplicitLod,
    Op::ImageSparseSampleDrefImplicitLod,
    Op::ImageSparseSampleProjImplicitLod,
    Op::ImageSparseSampleProjDrefImplicitLod,
];

/// Sample instructions that use explicit LOD.
const EXPLICIT_LOD_OPS: &[Op] = &[
    Op::ImageSampleExplicitLod,
    Op::ImageSampleDrefExplicitLod,
    Op::ImageSampleProjExplicitLod,
    Op::ImageSampleProjDrefExplicitLod,
    Op::ImageSparseSampleExplicitLod,
    Op::ImageSparseSampleDrefExplicitLod,
    Op::ImageSparseSampleProjExplicitLod,
    Op::ImageSparseSampleProjDrefExplicitLod,
];

/// Projection sample operations.
#[allow(dead_code)]
const PROJ_OPS: &[Op] = &[
    Op::ImageSampleProjImplicitLod,
    Op::ImageSampleProjDrefImplicitLod,
    Op::ImageSparseSampleProjImplicitLod,
    Op::ImageSparseSampleProjDrefImplicitLod,
    Op::ImageSampleProjExplicitLod,
    Op::ImageSampleProjDrefExplicitLod,
    Op::ImageSparseSampleProjExplicitLod,
    Op::ImageSparseSampleProjDrefExplicitLod,
];

/// Gather operations.
const GATHER_OPS: &[Op] = &[
    Op::ImageGather,
    Op::ImageDrefGather,
    Op::ImageSparseGather,
    Op::ImageSparseDrefGather,
];

/// Fetch operations.
const FETCH_OPS: &[Op] = &[Op::ImageFetch, Op::ImageSparseFetch];

/// Read/write operations.
const READ_WRITE_OPS: &[Op] = &[
    Op::ImageRead,
    Op::ImageWrite,
    Op::ImageSparseRead,
];

/// Query operations.
const QUERY_OPS: &[Op] = &[
    Op::ImageQueryFormat,
    Op::ImageQueryOrder,
    Op::ImageQuerySizeLod,
    Op::ImageQuerySize,
    Op::ImageQueryLod,
    Op::ImageQueryLevels,
    Op::ImageQuerySamples,
];

/// All image operations that need validation.
const ALL_IMAGE_OPS: &[Op] = &[
    // Sampling
    Op::ImageSampleImplicitLod,
    Op::ImageSampleExplicitLod,
    Op::ImageSampleDrefImplicitLod,
    Op::ImageSampleDrefExplicitLod,
    Op::ImageSampleProjImplicitLod,
    Op::ImageSampleProjExplicitLod,
    Op::ImageSampleProjDrefImplicitLod,
    Op::ImageSampleProjDrefExplicitLod,
    // Sparse sampling
    Op::ImageSparseSampleImplicitLod,
    Op::ImageSparseSampleExplicitLod,
    Op::ImageSparseSampleDrefImplicitLod,
    Op::ImageSparseSampleDrefExplicitLod,
    Op::ImageSparseSampleProjImplicitLod,
    Op::ImageSparseSampleProjExplicitLod,
    Op::ImageSparseSampleProjDrefImplicitLod,
    Op::ImageSparseSampleProjDrefExplicitLod,
    // Fetch
    Op::ImageFetch,
    Op::ImageSparseFetch,
    // Gather
    Op::ImageGather,
    Op::ImageDrefGather,
    Op::ImageSparseGather,
    Op::ImageSparseDrefGather,
    // Read/Write
    Op::ImageRead,
    Op::ImageWrite,
    Op::ImageSparseRead,
    // Query
    Op::ImageQueryFormat,
    Op::ImageQueryOrder,
    Op::ImageQuerySizeLod,
    Op::ImageQuerySize,
    Op::ImageQueryLod,
    Op::ImageQueryLevels,
    Op::ImageQuerySamples,
    // Texel
    Op::ImageTexelPointer,
    // Sampled image
    Op::SampledImage,
    Op::Image,
];

fn is_implicit_lod(op: Op) -> bool {
    IMPLICIT_LOD_OPS.contains(&op)
}

fn is_explicit_lod(op: Op) -> bool {
    EXPLICIT_LOD_OPS.contains(&op)
}

#[allow(dead_code)]
fn is_proj(op: Op) -> bool {
    PROJ_OPS.contains(&op)
}

fn is_gather(op: Op) -> bool {
    GATHER_OPS.contains(&op)
}

fn is_fetch(op: Op) -> bool {
    FETCH_OPS.contains(&op)
}

fn is_read_write(op: Op) -> bool {
    READ_WRITE_OPS.contains(&op)
}

fn is_query(op: Op) -> bool {
    QUERY_OPS.contains(&op)
}

fn is_image_op(op: Op) -> bool {
    ALL_IMAGE_OPS.contains(&op)
}

// ============================================================================
// Coordinate Size Calculation
// ============================================================================

/// Get the number of coordinate components for a single plane.
#[allow(dead_code)]
fn get_plane_coord_size(info: &ImageTypeInfo) -> u32 {
    match info.dim {
        Dim::Dim1D | Dim::DimBuffer => 1,
        Dim::Dim2D | Dim::DimRect | Dim::DimSubpassData | Dim::DimTileImageDataEXT => 2,
        Dim::Dim3D | Dim::DimCube => 3,
    }
}

/// Get the minimum coordinate size for an image operation.
#[allow(dead_code)]
fn get_min_coord_size(op: Op, info: &ImageTypeInfo) -> u32 {
    // Read/Write on Cube use UV (2D), not direction vector
    if info.dim == Dim::DimCube && is_read_write(op) {
        return 3;
    }
    get_plane_coord_size(info) + info.arrayed + if is_proj(op) { 1 } else { 0 }
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
                    if !is_image_op(inst.class.opcode) {
                        continue;
                    }

                    // Skip query operations - they don't have image operands
                    if is_query(inst.class.opcode) {
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
                        if !is_implicit_lod(inst.class.opcode) {
                            return Err(ValidationError::ImageOperandBiasRequiresImplicitLod {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate Lod operand
                    if mask.contains(ImageOperands::LOD) {
                        let valid_for_lod = is_explicit_lod(inst.class.opcode)
                            || is_fetch(inst.class.opcode)
                            || (is_read_write(inst.class.opcode)
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
                        if !is_explicit_lod(inst.class.opcode) {
                            return Err(ValidationError::ImageOperandGradRequiresExplicitLod {
                                function: function_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            });
                        }
                    }

                    // Validate ConstOffsets operand
                    if mask.contains(ImageOperands::CONST_OFFSETS) {
                        if !is_gather(inst.class.opcode) {
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
                    .any(|i| is_implicit_lod(i.class.opcode))
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
                        if is_implicit_lod(inst.class.opcode) {
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

// ============================================================================
// All Image Rules
// ============================================================================

/// Returns all image validation rules.
pub fn all_image_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ImageTypeRule,
        &ImageOperandRule,
        &ImageSampleExecutionModelRule,
        &ImageReadWriteRule,
        &ImageQueryRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_implicit_lod() {
        assert!(is_implicit_lod(Op::ImageSampleImplicitLod));
        assert!(is_implicit_lod(Op::ImageSampleDrefImplicitLod));
        assert!(!is_implicit_lod(Op::ImageSampleExplicitLod));
        assert!(!is_implicit_lod(Op::ImageFetch));
    }

    #[test]
    fn test_is_explicit_lod() {
        assert!(is_explicit_lod(Op::ImageSampleExplicitLod));
        assert!(is_explicit_lod(Op::ImageSampleDrefExplicitLod));
        assert!(!is_explicit_lod(Op::ImageSampleImplicitLod));
        assert!(!is_explicit_lod(Op::ImageFetch));
    }

    #[test]
    fn test_is_proj() {
        assert!(is_proj(Op::ImageSampleProjImplicitLod));
        assert!(is_proj(Op::ImageSampleProjExplicitLod));
        assert!(!is_proj(Op::ImageSampleImplicitLod));
        assert!(!is_proj(Op::ImageFetch));
    }

    #[test]
    fn test_is_gather() {
        assert!(is_gather(Op::ImageGather));
        assert!(is_gather(Op::ImageDrefGather));
        assert!(!is_gather(Op::ImageFetch));
        assert!(!is_gather(Op::ImageSampleImplicitLod));
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
