//! Mode setting validation rules.
//!
//! This module validates SPIR-V mode setting requirements including:
//!
//! - Entry point validation (function, return type, parameters)
//! - Execution mode constraints (fragment origin, tessellation, geometry)
//! - Memory model validation
//! - Capability dependencies
//! - Duplicate execution mode detection
//! - LocalSize validation
//! - Execution mode to execution model compatibility

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, MemoryModel,
    Op,
};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::Id;
use crate::validation::ValidationResult;

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Validates entry point requirements.
///
/// - Entry point must be a function
/// - Entry point must return void
/// - Non-Kernel entry points must have zero parameters
pub struct EntryPointValidationRule;

impl ValidationRule for EntryPointValidationRule {
    fn name(&self) -> &'static str {
        "entry-point-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build map of function ID -> function type ID
        let mut function_types: HashMap<u32, u32> = HashMap::new();
        for func in &module.functions {
            if let Some(def) = &func.def {
                if let (Some(func_id), Some(Operand::IdRef(type_id))) =
                    (def.result_id, def.operands.get(1))
                {
                    function_types.insert(func_id, *type_id);
                }
            }
        }

        // Build map of type ID -> type instruction for type functions
        let mut type_functions: HashMap<u32, (Option<u32>, usize)> = HashMap::new();
        for inst in &module.types_global_values {
            if inst.class.opcode == Op::TypeFunction {
                if let Some(type_id) = inst.result_id {
                    // TypeFunction: result_id, return_type, param_types...
                    let return_type = inst.operands.first().and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });
                    let param_count = inst.operands.len().saturating_sub(1);
                    type_functions.insert(type_id, (return_type, param_count));
                }
            }
        }

        // Build set of void type IDs
        let void_types: HashSet<u32> = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::TypeVoid)
            .filter_map(|inst| inst.result_id)
            .collect();

        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }

            let execution_model = ep.operands.first().and_then(|op| match op {
                Operand::ExecutionModel(model) => Some(*model),
                _ => None,
            });

            let Some(Operand::IdRef(func_id)) = ep.operands.get(1) else {
                continue;
            };

            // Check that entry point is a function
            let Some(func_type_id) = function_types.get(func_id) else {
                return Err(ValidationError::EntryPointNotFunction {
                    entry_point: to_id(*func_id),
                }
                .into());
            };

            // Get function type info
            let Some((return_type, param_count)) = type_functions.get(func_type_id) else {
                continue;
            };

            // Check return type is void
            if let Some(ret_type) = return_type {
                if !void_types.contains(ret_type) {
                    return Err(ValidationError::EntryPointReturnTypeNotVoid {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                }
            }

            // Non-Kernel entry points must have zero parameters
            if execution_model != Some(ExecutionModel::Kernel) && *param_count > 0 {
                return Err(ValidationError::EntryPointNonZeroParameters {
                    entry_point: to_id(*func_id),
                    param_count: *param_count as u32,
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Validates fragment shader execution mode requirements.
///
/// - Fragment must have exactly one of OriginUpperLeft/OriginLowerLeft
/// - At most one depth mode (DepthGreater, DepthLess, DepthUnchanged)
/// - At most one interlock mode
pub struct FragmentExecutionModeRule;

impl ValidationRule for FragmentExecutionModeRule {
    fn name(&self) -> &'static str {
        "fragment-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.declared_capabilities.contains(&Capability::Shader) {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of entry point -> execution model
        let mut entry_point_models: HashMap<u32, ExecutionModel> = HashMap::new();
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            if let (Some(Operand::ExecutionModel(model)), Some(Operand::IdRef(func_id))) =
                (ep.operands.first(), ep.operands.get(1))
            {
                entry_point_models.insert(*func_id, *model);
            }
        }

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check fragment entry points
        for (func_id, model) in &entry_point_models {
            if *model != ExecutionModel::Fragment {
                continue;
            }

            let modes = entry_point_modes.get(func_id);

            // Check origin modes
            let has_upper = modes.is_some_and(|m| m.contains(&ExecutionMode::OriginUpperLeft));
            let has_lower = modes.is_some_and(|m| m.contains(&ExecutionMode::OriginLowerLeft));

            if has_upper && has_lower {
                return Err(ValidationError::FragmentMultipleOriginModes {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            if !has_upper && !has_lower {
                return Err(ValidationError::FragmentMissingOriginMode {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            // Check depth modes
            if let Some(modes) = modes {
                let depth_count = [
                    ExecutionMode::DepthGreater,
                    ExecutionMode::DepthLess,
                    ExecutionMode::DepthUnchanged,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if depth_count > 1 {
                    return Err(ValidationError::FragmentMultipleDepthModes {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                }

                // Check interlock modes
                let interlock_count = [
                    ExecutionMode::PixelInterlockOrderedEXT,
                    ExecutionMode::PixelInterlockUnorderedEXT,
                    ExecutionMode::SampleInterlockOrderedEXT,
                    ExecutionMode::SampleInterlockUnorderedEXT,
                    ExecutionMode::ShadingRateInterlockOrderedEXT,
                    ExecutionMode::ShadingRateInterlockUnorderedEXT,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if interlock_count > 1 {
                    return Err(ValidationError::FragmentMultipleInterlockModes {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                }

                // Check AMD stencil ref front modes
                let stencil_front_count = [
                    ExecutionMode::StencilRefUnchangedFrontAMD,
                    ExecutionMode::StencilRefLessFrontAMD,
                    ExecutionMode::StencilRefGreaterFrontAMD,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if stencil_front_count > 1 {
                    return Err(ValidationError::FragmentMultipleStencilRefFrontModes {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                }

                // Check AMD stencil ref back modes
                let stencil_back_count = [
                    ExecutionMode::StencilRefUnchangedBackAMD,
                    ExecutionMode::StencilRefLessBackAMD,
                    ExecutionMode::StencilRefGreaterBackAMD,
                ]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

                if stencil_back_count > 1 {
                    return Err(ValidationError::FragmentMultipleStencilRefBackModes {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

/// Validates tessellation shader execution mode requirements.
///
/// - At most one spacing mode (SpacingEqual, SpacingFractionalEven/Odd)
/// - At most one primitive type (Triangles, Quads, Isolines)
/// - At most one vertex order (VertexOrderCw, VertexOrderCcw)
pub struct TessellationExecutionModeRule;

impl ValidationRule for TessellationExecutionModeRule {
    fn name(&self) -> &'static str {
        "tessellation-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.declared_capabilities.contains(&Capability::Shader) {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of entry point -> execution model
        let mut entry_point_models: HashMap<u32, ExecutionModel> = HashMap::new();
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            if let (Some(Operand::ExecutionModel(model)), Some(Operand::IdRef(func_id))) =
                (ep.operands.first(), ep.operands.get(1))
            {
                entry_point_models.insert(*func_id, *model);
            }
        }

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check tessellation entry points
        for (func_id, model) in &entry_point_models {
            if *model != ExecutionModel::TessellationControl
                && *model != ExecutionModel::TessellationEvaluation
            {
                continue;
            }

            let Some(modes) = entry_point_modes.get(func_id) else {
                continue;
            };

            // Check spacing modes
            let spacing_count = [
                ExecutionMode::SpacingEqual,
                ExecutionMode::SpacingFractionalEven,
                ExecutionMode::SpacingFractionalOdd,
            ]
            .iter()
            .filter(|m| modes.contains(m))
            .count();

            if spacing_count > 1 {
                return Err(ValidationError::TessellationMultipleSpacingModes {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            // Check primitive types
            let primitive_count = [
                ExecutionMode::Triangles,
                ExecutionMode::Quads,
                ExecutionMode::Isolines,
            ]
            .iter()
            .filter(|m| modes.contains(m))
            .count();

            if primitive_count > 1 {
                return Err(ValidationError::TessellationMultiplePrimitiveTypes {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            // Check vertex order modes
            let vertex_order_count = [ExecutionMode::VertexOrderCw, ExecutionMode::VertexOrderCcw]
                .iter()
                .filter(|m| modes.contains(m))
                .count();

            if vertex_order_count > 1 {
                return Err(ValidationError::TessellationMultipleVertexOrderModes {
                    entry_point: to_id(*func_id),
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Validates geometry shader execution mode requirements.
///
/// - Exactly one input primitive type (InputPoints, InputLines, etc.)
/// - Exactly one output primitive type (OutputPoints, OutputLineStrip, OutputTriangleStrip)
pub struct GeometryExecutionModeRule;

impl ValidationRule for GeometryExecutionModeRule {
    fn name(&self) -> &'static str {
        "geometry-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.declared_capabilities.contains(&Capability::Shader) {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of entry point -> execution model
        let mut entry_point_models: HashMap<u32, ExecutionModel> = HashMap::new();
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            if let (Some(Operand::ExecutionModel(model)), Some(Operand::IdRef(func_id))) =
                (ep.operands.first(), ep.operands.get(1))
            {
                entry_point_models.insert(*func_id, *model);
            }
        }

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check geometry entry points
        for (func_id, model) in &entry_point_models {
            if *model != ExecutionModel::Geometry {
                continue;
            }

            let modes = entry_point_modes.get(func_id);

            // Check input primitive types - exactly one required
            let input_count = [
                ExecutionMode::InputPoints,
                ExecutionMode::InputLines,
                ExecutionMode::InputLinesAdjacency,
                ExecutionMode::Triangles,
                ExecutionMode::InputTrianglesAdjacency,
            ]
            .iter()
            .filter(|m| modes.is_some_and(|s| s.contains(m)))
            .count();

            if input_count != 1 {
                return Err(ValidationError::GeometryMissingInputPrimitiveType {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            // Check output primitive types - exactly one required
            let output_count = [
                ExecutionMode::OutputPoints,
                ExecutionMode::OutputLineStrip,
                ExecutionMode::OutputTriangleStrip,
            ]
            .iter()
            .filter(|m| modes.is_some_and(|s| s.contains(m)))
            .count();

            if output_count != 1 {
                return Err(ValidationError::GeometryMissingOutputPrimitiveType {
                    entry_point: to_id(*func_id),
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Validates MeshEXT shader execution mode requirements.
///
/// - Exactly one output primitive type (OutputPoints, OutputLinesEXT, OutputTrianglesEXT)
/// - Both OutputPrimitivesEXT and OutputVertices must be specified
pub struct MeshExtExecutionModeRule;

impl ValidationRule for MeshExtExecutionModeRule {
    fn name(&self) -> &'static str {
        "mesh-ext-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.declared_capabilities.contains(&Capability::Shader) {
            return Ok(());
        }

        let module = ctx.module();

        // Build map of entry point -> execution model
        let mut entry_point_models: HashMap<u32, ExecutionModel> = HashMap::new();
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            if let (Some(Operand::ExecutionModel(model)), Some(Operand::IdRef(func_id))) =
                (ep.operands.first(), ep.operands.get(1))
            {
                entry_point_models.insert(*func_id, *model);
            }
        }

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check MeshEXT entry points
        for (func_id, model) in &entry_point_models {
            if *model != ExecutionModel::MeshEXT {
                continue;
            }

            let modes = entry_point_modes.get(func_id);

            // Check output primitive types - exactly one required
            let output_prim_count = [
                ExecutionMode::OutputPoints,
                ExecutionMode::OutputLinesEXT,
                ExecutionMode::OutputTrianglesEXT,
            ]
            .iter()
            .filter(|m| modes.is_some_and(|s| s.contains(m)))
            .count();

            if output_prim_count != 1 {
                return Err(ValidationError::MeshExtMissingOutputPrimitiveType {
                    entry_point: to_id(*func_id),
                }
                .into());
            }

            // Check that both OutputPrimitivesEXT and OutputVertices are specified
            let has_output_prims =
                modes.is_some_and(|s| s.contains(&ExecutionMode::OutputPrimitivesEXT));
            let has_output_verts =
                modes.is_some_and(|s| s.contains(&ExecutionMode::OutputVertices));

            if !has_output_prims || !has_output_verts {
                return Err(ValidationError::MeshExtMissingOutputModes {
                    entry_point: to_id(*func_id),
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Validates GLCompute LocalSize requirements in Vulkan.
///
/// In Vulkan, GLCompute entry points require LocalSize, LocalSizeId, or WorkgroupSize.
pub struct VulkanGLComputeLocalSizeRule;

impl ValidationRule for VulkanGLComputeLocalSizeRule {
    fn name(&self) -> &'static str {
        "vulkan-glcompute-localsize"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module();

        // Check for WorkgroupSize decoration
        let mut has_workgroup_size = false;
        for inst in &module.annotations {
            if inst.class.opcode == Op::Decorate {
                if let (
                    Some(Operand::Decoration(Decoration::BuiltIn)),
                    Some(Operand::BuiltIn(bi)),
                ) = (inst.operands.get(1), inst.operands.get(2))
                {
                    if *bi == BuiltIn::WorkgroupSize {
                        has_workgroup_size = true;
                        break;
                    }
                }
            }
        }

        // Build map of entry point -> execution modes with values
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        let mut has_local_size_id = false;
        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };
            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);

            if mode.class.opcode == Op::ExecutionModeId && *exec_mode == ExecutionMode::LocalSizeId
            {
                has_local_size_id = true;
            }
        }

        // Check GLCompute and Mesh/Task entry points for LocalSize requirement
        for ep in &module.entry_points {
            if ep.class.opcode != Op::EntryPoint {
                continue;
            }
            let Some(Operand::ExecutionModel(model)) = ep.operands.first() else {
                continue;
            };

            let requires_local_size = matches!(
                model,
                ExecutionModel::GLCompute
                    | ExecutionModel::MeshEXT
                    | ExecutionModel::MeshNV
                    | ExecutionModel::TaskEXT
                    | ExecutionModel::TaskNV
            );
            if !requires_local_size {
                continue;
            }

            let Some(Operand::IdRef(func_id)) = ep.operands.get(1) else {
                continue;
            };

            let modes = entry_point_modes.get(func_id);
            let has_local_size = modes.is_some_and(|s| s.contains(&ExecutionMode::LocalSize));

            // For TileShadingQCOM capability, TileShadingRateQCOM mode is also acceptable
            let has_tile_shading = ctx
                .declared_capabilities
                .contains(&Capability::TileShadingQCOM)
                && modes.is_some_and(|s| s.contains(&ExecutionMode::TileShadingRateQCOM));

            if !has_local_size && !has_workgroup_size && !has_local_size_id && !has_tile_shading {
                if *model == ExecutionModel::GLCompute {
                    return Err(ValidationError::VulkanGLComputeMissingLocalSize {
                        entry_point: to_id(*func_id),
                    }
                    .into());
                } else {
                    return Err(ValidationError::MissingLocalSizeForModel {
                        entry_point: to_id(*func_id),
                        execution_model: *model,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

/// Validates LocalSize execution mode constraints.
///
/// - Product of X, Y, Z must not be zero
/// - DerivativeGroupQuadsKHR requires X and Y to be multiples of 2
/// - DerivativeGroupLinearKHR requires product to be a multiple of 4
pub struct LocalSizeValidationRule;

impl ValidationRule for LocalSizeValidationRule {
    fn name(&self) -> &'static str {
        "localsize-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();
        let mut local_size_values: HashMap<u32, (u32, u32, u32)> = HashMap::new();

        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);

            // Capture LocalSize values
            if *exec_mode == ExecutionMode::LocalSize {
                if let (
                    Some(Operand::LiteralBit32(x)),
                    Some(Operand::LiteralBit32(y)),
                    Some(Operand::LiteralBit32(z)),
                ) = (
                    mode.operands.get(2),
                    mode.operands.get(3),
                    mode.operands.get(4),
                ) {
                    local_size_values.insert(*func_id, (*x, *y, *z));
                }
            }
        }

        // Validate LocalSize for each entry point
        for (func_id, (x, y, z)) in &local_size_values {
            let product = (*x as u64) * (*y as u64) * (*z as u64);

            // Product must not be zero
            if product == 0 {
                return Err(ValidationError::LocalSizeProductZero {
                    x: *x,
                    y: *y,
                    z: *z,
                }
                .into());
            }

            let modes = entry_point_modes.get(func_id);

            // Check DerivativeGroupQuadsKHR
            if modes.is_some_and(|s| s.contains(&ExecutionMode::DerivativeGroupQuadsKHR))
                && (*x % 2 != 0 || *y % 2 != 0)
            {
                return Err(ValidationError::DerivativeGroupQuadsRequiresMultipleOf2 {
                    x: *x as u64,
                    y: *y as u64,
                }
                .into());
            }

            // Check DerivativeGroupLinearKHR
            if modes.is_some_and(|s| s.contains(&ExecutionMode::DerivativeGroupLinearKHR))
                && !product.is_multiple_of(4)
            {
                return Err(
                    ValidationError::DerivativeGroupLinearRequiresMultipleOf4 { product }.into(),
                );
            }
        }

        Ok(())
    }
}

/// Validates FPFastMathDefault execution mode conflicts.
///
/// - Cannot be combined with ContractionOff
/// - Cannot be combined with SignedZeroInfNanPreserve
pub struct FPFastMathDefaultConflictsRule;

impl ValidationRule for FPFastMathDefaultConflictsRule {
    fn name(&self) -> &'static str {
        "fp-fast-math-default-conflicts"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build map of entry point -> execution modes
        let mut entry_point_modes: HashMap<u32, HashSet<ExecutionMode>> = HashMap::new();

        for mode in &module.execution_modes {
            let Some(Operand::IdRef(func_id)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            entry_point_modes
                .entry(*func_id)
                .or_default()
                .insert(*exec_mode);
        }

        // Check for conflicts
        for modes in entry_point_modes.values() {
            if modes.contains(&ExecutionMode::FPFastMathDefault) {
                if modes.contains(&ExecutionMode::ContractionOff) {
                    return Err(
                        ValidationError::FPFastMathDefaultConflictsWithContractionOff.into(),
                    );
                }
                if modes.contains(&ExecutionMode::SignedZeroInfNanPreserve) {
                    return Err(
                        ValidationError::FPFastMathDefaultConflictsWithSignedZeroInfNanPreserve
                            .into(),
                    );
                }
            }
        }

        Ok(())
    }
}

/// Validates TileShadingRateQCOM execution mode.
///
/// In Vulkan, the x and y values must be powers of 2.
pub struct TileShadingRateQCOMRule;

impl ValidationRule for TileShadingRateQCOMRule {
    fn name(&self) -> &'static str {
        "tile-shading-rate-qcom"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module();

        for mode in &module.execution_modes {
            let Some(Operand::ExecutionMode(ExecutionMode::TileShadingRateQCOM)) =
                mode.operands.get(1)
            else {
                continue;
            };

            // Get x and y values
            let x = mode
                .operands
                .get(2)
                .and_then(|op| match op {
                    Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                })
                .unwrap_or(0);

            let y = mode
                .operands
                .get(3)
                .and_then(|op| match op {
                    Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                })
                .unwrap_or(0);

            // Check that x and y are powers of 2
            let is_power_of_2 = |n: u32| n > 0 && (n & (n - 1)) == 0;

            if !is_power_of_2(x) || !is_power_of_2(y) {
                return Err(ValidationError::TileShadingRateQCOMNotPowerOf2.into());
            }
        }

        Ok(())
    }
}

/// Validates Vulkan-specific execution mode restrictions.
///
/// - OriginLowerLeft is not allowed in Vulkan
/// - PixelCenterInteger is not allowed in Vulkan
pub struct VulkanExecutionModeRule;

impl ValidationRule for VulkanExecutionModeRule {
    fn name(&self) -> &'static str {
        "vulkan-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module();

        for mode in &module.execution_modes {
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            match exec_mode {
                ExecutionMode::OriginLowerLeft => {
                    return Err(ValidationError::VulkanOriginLowerLeftNotAllowed.into());
                }
                ExecutionMode::PixelCenterInteger => {
                    return Err(ValidationError::VulkanPixelCenterIntegerNotAllowed.into());
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Validates memory model requirements.
///
/// - VulkanMemoryModelKHR capability requires VulkanKHR memory model
/// - OpenCL requires Physical32/Physical64 addressing and OpenCL memory model
/// - Vulkan requires Logical or PhysicalStorageBuffer64 addressing
pub struct MemoryModelValidationRule;

impl ValidationRule for MemoryModelValidationRule {
    fn name(&self) -> &'static str {
        "memory-model-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Find memory model instruction
        let mut memory_model: Option<MemoryModel> = None;
        let mut addressing_model: Option<AddressingModel> = None;

        if let Some(inst) = &module.memory_model {
            if inst.class.opcode == Op::MemoryModel {
                if let Some(Operand::AddressingModel(addr)) = inst.operands.first() {
                    addressing_model = Some(*addr);
                }
                if let Some(Operand::MemoryModel(mem)) = inst.operands.get(1) {
                    memory_model = Some(*mem);
                }
            }
        }

        // VulkanMemoryModelKHR requires VulkanKHR memory model
        if ctx
            .declared_capabilities
            .contains(&Capability::VulkanMemoryModelKHR)
            && memory_model != Some(MemoryModel::VulkanKHR)
        {
            return Err(ValidationError::VulkanMemoryModelCapabilityRequiresVulkanKHR.into());
        }

        // Vulkan environment restrictions
        if ctx.env.is_vulkan() {
            if let Some(addr) = addressing_model {
                if addr != AddressingModel::Logical
                    && addr != AddressingModel::PhysicalStorageBuffer64
                {
                    return Err(ValidationError::VulkanInvalidAddressingModel {
                        addressing_model: addr,
                    }
                    .into());
                }
            }
        }

        // OpenCL environment restrictions
        if ctx.env.is_opencl() {
            if let Some(addr) = addressing_model {
                if addr != AddressingModel::Physical32 && addr != AddressingModel::Physical64 {
                    return Err(ValidationError::OpenCLInvalidAddressingModel {
                        addressing_model: addr,
                    }
                    .into());
                }
            }
            if let Some(mem) = memory_model {
                if mem != MemoryModel::OpenCL {
                    return Err(
                        ValidationError::OpenCLInvalidMemoryModel { memory_model: mem }.into(),
                    );
                }
            }
        }

        Ok(())
    }
}

/// Validates capability dependencies.
///
/// - CooperativeMatrixKHR with Shader requires VulkanMemoryModel
pub struct CapabilityDependenciesRule;

impl ValidationRule for CapabilityDependenciesRule {
    fn name(&self) -> &'static str {
        "capability-dependencies"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // CooperativeMatrixKHR + Shader requires VulkanMemoryModel
        if ctx
            .declared_capabilities
            .contains(&Capability::CooperativeMatrixKHR)
            && ctx.declared_capabilities.contains(&Capability::Shader)
            && !ctx
                .declared_capabilities
                .contains(&Capability::VulkanMemoryModel)
        {
            return Err(ValidationError::CooperativeMatrixRequiresVulkanMemoryModel.into());
        }

        Ok(())
    }
}

/// Validates that execution modes are not duplicated.
///
/// Most execution modes can only appear once per entry point.
pub struct DuplicateExecutionModeRule;

impl ValidationRule for DuplicateExecutionModeRule {
    fn name(&self) -> &'static str {
        "duplicate-execution-mode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Modes that can be specified multiple times per entry point
        let per_operand_modes = [
            ExecutionMode::DenormPreserve,
            ExecutionMode::DenormFlushToZero,
            ExecutionMode::SignedZeroInfNanPreserve,
            ExecutionMode::RoundingModeRTE,
            ExecutionMode::RoundingModeRTZ,
            ExecutionMode::FPFastMathDefault,
            ExecutionMode::RoundingModeRTPINTEL,
            ExecutionMode::RoundingModeRTNINTEL,
            ExecutionMode::FloatingPointModeALTINTEL,
            ExecutionMode::FloatingPointModeIEEEINTEL,
        ];

        // Track seen modes: (entry_point, mode) for per-entry modes
        // (entry_point, mode, operand) for per-operand modes
        let mut seen_per_entry: HashSet<(u32, ExecutionMode)> = HashSet::new();
        let mut seen_per_operand: HashSet<(u32, ExecutionMode, u32)> = HashSet::new();

        for mode in &module.execution_modes {
            let Some(Operand::IdRef(entry_point)) = mode.operands.first() else {
                continue;
            };
            let Some(Operand::ExecutionMode(exec_mode)) = mode.operands.get(1) else {
                continue;
            };

            if per_operand_modes.contains(exec_mode) {
                // Per-operand modes - check with operand
                let operand = mode
                    .operands
                    .get(2)
                    .and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(0);

                if !seen_per_operand.insert((*entry_point, *exec_mode, operand)) {
                    return Err(ValidationError::DuplicateExecutionMode {
                        entry_point: to_id(*entry_point),
                        mode: *exec_mode,
                    }
                    .into());
                }
            } else {
                // Per-entry modes
                if !seen_per_entry.insert((*entry_point, *exec_mode)) {
                    return Err(ValidationError::DuplicateExecutionMode {
                        entry_point: to_id(*entry_point),
                        mode: *exec_mode,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

/// Validates `OpSamplerImageAddressingModeNV` instruction requirements.
///
/// From validate_instruction.cpp:
/// - Requires `BindlessTextureNV` capability
/// - Must only be provided once
/// - Bit width must be 32 or 64
pub struct SamplerImageAddressingModeNVRule;

impl ValidationRule for SamplerImageAddressingModeNVRule {
    fn name(&self) -> &'static str {
        "sampler-image-addressing-mode-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        let mut seen = false;

        for inst in module.all_inst_iter() {
            if inst.class.opcode != Op::SamplerImageAddressingModeNV {
                continue;
            }

            // Requires BindlessTextureNV capability
            if !ctx.has_capability(Capability::BindlessTextureNV) {
                return Err(
                    ValidationError::SamplerImageAddressingModeNVRequiresBindlessTextureNV.into(),
                );
            }

            // Must only be provided once
            if seen {
                return Err(ValidationError::DuplicateSamplerImageAddressingMode.into());
            }
            seen = true;

            // Bit width must be 32 or 64
            let bit_width = inst
                .operands
                .first()
                .and_then(|op| match op {
                    Operand::LiteralBit32(v) => Some(*v),
                    _ => None,
                })
                .unwrap_or(0);

            if bit_width != 32 && bit_width != 64 {
                return Err(ValidationError::InvalidSamplerImageAddressingModeBitWidth {
                    bit_width,
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Returns all mode setting validation rules.
pub fn all_mode_setting_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(EntryPointValidationRule),
        Box::new(FragmentExecutionModeRule),
        Box::new(TessellationExecutionModeRule),
        Box::new(GeometryExecutionModeRule),
        Box::new(MeshExtExecutionModeRule),
        Box::new(VulkanExecutionModeRule),
        Box::new(VulkanGLComputeLocalSizeRule),
        Box::new(LocalSizeValidationRule),
        Box::new(MemoryModelValidationRule),
        Box::new(CapabilityDependenciesRule),
        Box::new(FPFastMathDefaultConflictsRule),
        Box::new(TileShadingRateQCOMRule),
        Box::new(DuplicateExecutionModeRule),
        Box::new(SamplerImageAddressingModeNVRule),
    ]
}
