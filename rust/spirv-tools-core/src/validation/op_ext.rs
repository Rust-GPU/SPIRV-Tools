//! Extension traits for SPIR-V opcode classification.
//!
//! This module provides extension traits on rspirv's `Op` enum to categorize
//! and query properties of SPIR-V opcodes for validation purposes.

use rspirv::spirv::{BuiltIn, Decoration, Op};

// ============================================================================
// Op Extension Trait
// ============================================================================

/// Extension trait for SPIR-V opcode classification.
///
/// Provides methods to categorize opcodes by their operational category,
/// which is useful for validation rules.
pub trait OpExt {
    // --- Image operation classification ---

    /// Returns true if this is a sample instruction using implicit LOD.
    fn is_implicit_lod(&self) -> bool;

    /// Returns true if this is a sample instruction using explicit LOD.
    fn is_explicit_lod(&self) -> bool;

    /// Returns true if this is a projection sample operation.
    fn is_proj(&self) -> bool;

    /// Returns true if this is a depth-reference (Dref) sample operation.
    fn is_dref(&self) -> bool;

    /// Returns true if this is a gather operation.
    fn is_gather(&self) -> bool;

    /// Returns true if this is a fetch operation.
    fn is_fetch(&self) -> bool;

    /// Returns true if this is an image read or write operation.
    fn is_image_read_write(&self) -> bool;

    /// Returns true if this is an image query operation.
    fn is_image_query(&self) -> bool;

    /// Returns true if this is any image operation that needs validation.
    fn is_image_op(&self) -> bool;

    /// Returns true if this is a sampling operation (implicit or explicit LOD).
    fn is_sample(&self) -> bool;

    // --- Other operation categories ---

    /// Returns true if this is an atomic operation.
    fn is_atomic(&self) -> bool;

    /// Returns true if this is a derivative operation.
    fn is_derivative(&self) -> bool;

    /// Returns true if this is a constant or undef instruction.
    fn is_constant_or_undef(&self) -> bool;

    /// Returns true if this is a spec constant instruction.
    fn is_spec_constant(&self) -> bool;

    /// Returns true if this is a scalar spec constant instruction.
    fn is_scalar_spec_constant(&self) -> bool;

    /// Returns true if this is any constant opcode.
    fn is_constant(&self) -> bool;

    /// Returns true if this is a composite type opcode.
    fn is_composite_type(&self) -> bool;

    /// Returns true if this is a scalar type opcode.
    fn is_scalar_type(&self) -> bool;

    /// Returns true if this is a barrier instruction.
    fn is_barrier(&self) -> bool;

    /// Returns true if this is a terminator instruction.
    fn is_terminator(&self) -> bool;

    /// Returns true if this is a merge instruction.
    fn is_merge(&self) -> bool;
}

impl OpExt for Op {
    fn is_implicit_lod(&self) -> bool {
        matches!(
            self,
            Op::ImageSampleImplicitLod
                | Op::ImageSampleDrefImplicitLod
                | Op::ImageSampleProjImplicitLod
                | Op::ImageSampleProjDrefImplicitLod
                | Op::ImageSparseSampleImplicitLod
                | Op::ImageSparseSampleDrefImplicitLod
                | Op::ImageSparseSampleProjImplicitLod
                | Op::ImageSparseSampleProjDrefImplicitLod
        )
    }

    fn is_explicit_lod(&self) -> bool {
        matches!(
            self,
            Op::ImageSampleExplicitLod
                | Op::ImageSampleDrefExplicitLod
                | Op::ImageSampleProjExplicitLod
                | Op::ImageSampleProjDrefExplicitLod
                | Op::ImageSparseSampleExplicitLod
                | Op::ImageSparseSampleDrefExplicitLod
                | Op::ImageSparseSampleProjExplicitLod
                | Op::ImageSparseSampleProjDrefExplicitLod
        )
    }

    fn is_proj(&self) -> bool {
        matches!(
            self,
            Op::ImageSampleProjImplicitLod
                | Op::ImageSampleProjDrefImplicitLod
                | Op::ImageSparseSampleProjImplicitLod
                | Op::ImageSparseSampleProjDrefImplicitLod
                | Op::ImageSampleProjExplicitLod
                | Op::ImageSampleProjDrefExplicitLod
                | Op::ImageSparseSampleProjExplicitLod
                | Op::ImageSparseSampleProjDrefExplicitLod
        )
    }

    fn is_dref(&self) -> bool {
        matches!(
            self,
            Op::ImageSampleDrefImplicitLod
                | Op::ImageSampleDrefExplicitLod
                | Op::ImageSampleProjDrefImplicitLod
                | Op::ImageSampleProjDrefExplicitLod
                | Op::ImageSparseSampleDrefImplicitLod
                | Op::ImageSparseSampleDrefExplicitLod
                | Op::ImageSparseSampleProjDrefImplicitLod
                | Op::ImageSparseSampleProjDrefExplicitLod
                | Op::ImageDrefGather
                | Op::ImageSparseDrefGather
        )
    }

    fn is_gather(&self) -> bool {
        matches!(
            self,
            Op::ImageGather
                | Op::ImageDrefGather
                | Op::ImageSparseGather
                | Op::ImageSparseDrefGather
        )
    }

    fn is_fetch(&self) -> bool {
        matches!(self, Op::ImageFetch | Op::ImageSparseFetch)
    }

    fn is_image_read_write(&self) -> bool {
        matches!(self, Op::ImageRead | Op::ImageWrite | Op::ImageSparseRead)
    }

    fn is_image_query(&self) -> bool {
        matches!(
            self,
            Op::ImageQueryFormat
                | Op::ImageQueryOrder
                | Op::ImageQuerySizeLod
                | Op::ImageQuerySize
                | Op::ImageQueryLod
                | Op::ImageQueryLevels
                | Op::ImageQuerySamples
        )
    }

    fn is_image_op(&self) -> bool {
        self.is_implicit_lod()
            || self.is_explicit_lod()
            || self.is_gather()
            || self.is_fetch()
            || self.is_image_read_write()
            || self.is_image_query()
            || matches!(self, Op::ImageTexelPointer | Op::SampledImage | Op::Image)
    }

    fn is_sample(&self) -> bool {
        self.is_implicit_lod() || self.is_explicit_lod()
    }

    fn is_atomic(&self) -> bool {
        matches!(
            self,
            Op::AtomicLoad
                | Op::AtomicStore
                | Op::AtomicExchange
                | Op::AtomicCompareExchange
                | Op::AtomicCompareExchangeWeak
                | Op::AtomicIIncrement
                | Op::AtomicIDecrement
                | Op::AtomicIAdd
                | Op::AtomicISub
                | Op::AtomicSMin
                | Op::AtomicUMin
                | Op::AtomicSMax
                | Op::AtomicUMax
                | Op::AtomicAnd
                | Op::AtomicOr
                | Op::AtomicXor
                | Op::AtomicFlagTestAndSet
                | Op::AtomicFlagClear
                | Op::AtomicFMinEXT
                | Op::AtomicFMaxEXT
                | Op::AtomicFAddEXT
        )
    }

    fn is_derivative(&self) -> bool {
        matches!(
            self,
            Op::DPdx
                | Op::DPdy
                | Op::Fwidth
                | Op::DPdxFine
                | Op::DPdyFine
                | Op::FwidthFine
                | Op::DPdxCoarse
                | Op::DPdyCoarse
                | Op::FwidthCoarse
        )
    }

    fn is_constant_or_undef(&self) -> bool {
        matches!(
            self,
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
                | Op::Undef
        )
    }

    fn is_spec_constant(&self) -> bool {
        matches!(
            self,
            Op::SpecConstant
                | Op::SpecConstantTrue
                | Op::SpecConstantFalse
                | Op::SpecConstantComposite
                | Op::SpecConstantOp
        )
    }

    fn is_scalar_spec_constant(&self) -> bool {
        matches!(
            self,
            Op::SpecConstant | Op::SpecConstantTrue | Op::SpecConstantFalse
        )
    }

    fn is_constant(&self) -> bool {
        matches!(
            self,
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

    fn is_composite_type(&self) -> bool {
        matches!(
            self,
            Op::TypeArray
                | Op::TypeRuntimeArray
                | Op::TypeStruct
                | Op::TypeVector
                | Op::TypeMatrix
                | Op::TypeCooperativeMatrixKHR
                | Op::TypeCooperativeMatrixNV
                | Op::TypeCooperativeVectorNV
        )
    }

    fn is_scalar_type(&self) -> bool {
        matches!(
            self,
            Op::TypeInt | Op::TypeFloat | Op::TypeBool | Op::TypePointer
        )
    }

    fn is_barrier(&self) -> bool {
        matches!(
            self,
            Op::ControlBarrier | Op::MemoryBarrier | Op::MemoryNamedBarrier
        )
    }

    fn is_terminator(&self) -> bool {
        matches!(
            self,
            Op::Branch
                | Op::BranchConditional
                | Op::Switch
                | Op::Kill
                | Op::Return
                | Op::ReturnValue
                | Op::Unreachable
                | Op::TerminateInvocation
                | Op::IgnoreIntersectionKHR
                | Op::TerminateRayKHR
                | Op::EmitMeshTasksEXT
        )
    }

    fn is_merge(&self) -> bool {
        matches!(self, Op::SelectionMerge | Op::LoopMerge)
    }
}

// ============================================================================
// Decoration Extension Trait
// ============================================================================

/// Extension trait for SPIR-V decoration classification.
pub trait DecorationExt {
    /// Returns true if this decoration can only be applied to struct members.
    fn is_member_only(&self) -> bool;

    /// Returns true if this decoration cannot be applied to struct members.
    fn is_non_member(&self) -> bool;
}

impl DecorationExt for Decoration {
    fn is_member_only(&self) -> bool {
        matches!(
            self,
            Decoration::RowMajor
                | Decoration::ColMajor
                | Decoration::MatrixStride
                | Decoration::Offset
        )
    }

    fn is_non_member(&self) -> bool {
        matches!(
            self,
            Decoration::Block
                | Decoration::BufferBlock
                | Decoration::Location
                | Decoration::Component
                | Decoration::Binding
                | Decoration::DescriptorSet
                | Decoration::InputAttachmentIndex
        )
    }
}

// ============================================================================
// BuiltIn Extension Trait
// ============================================================================

/// Extension trait for SPIR-V built-in classification.
pub trait BuiltInExt {
    /// Returns true if this built-in is only valid in Fragment shaders.
    fn is_fragment_only(&self) -> bool;

    /// Returns true if this built-in is a barycentric coordinate.
    fn is_barycentric(&self) -> bool;

    /// Returns true if this built-in is a mesh shader output.
    fn is_mesh_output(&self) -> bool;

    /// Returns true if this built-in is only valid in compute shaders.
    fn is_compute_only(&self) -> bool;

    /// Returns true if this built-in is only valid in Kernel execution model.
    fn is_kernel_only(&self) -> bool;

    /// Returns true if this built-in is a ray tracing built-in.
    fn is_ray_tracing(&self) -> bool;

    /// Returns true if this built-in requires Input storage class.
    fn requires_input_storage_class(&self) -> bool;

    /// Returns true if this built-in requires Output storage class.
    fn requires_output_storage_class(&self) -> bool;
}

impl BuiltInExt for BuiltIn {
    fn is_fragment_only(&self) -> bool {
        matches!(
            self,
            BuiltIn::FragCoord
                | BuiltIn::PointCoord
                | BuiltIn::FrontFacing
                | BuiltIn::SampleId
                | BuiltIn::SamplePosition
                | BuiltIn::SampleMask
                | BuiltIn::FragDepth
                | BuiltIn::HelperInvocation
                | BuiltIn::FragInvocationCountEXT
                | BuiltIn::FragSizeEXT
                | BuiltIn::FragStencilRefEXT
                | BuiltIn::FullyCoveredEXT
                | BuiltIn::BaryCoordKHR
                | BuiltIn::BaryCoordNoPerspKHR
                | BuiltIn::BaryCoordSmoothAMD
                | BuiltIn::BaryCoordSmoothCentroidAMD
                | BuiltIn::BaryCoordSmoothSampleAMD
                | BuiltIn::BaryCoordNoPerspAMD
                | BuiltIn::BaryCoordNoPerspCentroidAMD
                | BuiltIn::BaryCoordNoPerspSampleAMD
                | BuiltIn::BaryCoordPullModelAMD
        )
    }

    fn is_barycentric(&self) -> bool {
        matches!(
            self,
            BuiltIn::BaryCoordKHR
                | BuiltIn::BaryCoordNoPerspKHR
                | BuiltIn::BaryCoordSmoothAMD
                | BuiltIn::BaryCoordSmoothCentroidAMD
                | BuiltIn::BaryCoordSmoothSampleAMD
                | BuiltIn::BaryCoordNoPerspAMD
                | BuiltIn::BaryCoordNoPerspCentroidAMD
                | BuiltIn::BaryCoordNoPerspSampleAMD
                | BuiltIn::BaryCoordPullModelAMD
        )
    }

    fn is_mesh_output(&self) -> bool {
        matches!(
            self,
            BuiltIn::PrimitivePointIndicesEXT
                | BuiltIn::PrimitiveLineIndicesEXT
                | BuiltIn::PrimitiveTriangleIndicesEXT
                | BuiltIn::CullPrimitiveEXT
        )
    }

    fn is_compute_only(&self) -> bool {
        matches!(
            self,
            BuiltIn::GlobalInvocationId
                | BuiltIn::LocalInvocationId
                | BuiltIn::LocalInvocationIndex
                | BuiltIn::NumWorkgroups
                | BuiltIn::WorkgroupId
                | BuiltIn::NumSubgroups
                | BuiltIn::SubgroupId
        )
    }

    fn is_kernel_only(&self) -> bool {
        matches!(
            self,
            BuiltIn::WorkDim
                | BuiltIn::GlobalSize
                | BuiltIn::GlobalOffset
                | BuiltIn::EnqueuedWorkgroupSize
                | BuiltIn::GlobalLinearId
                | BuiltIn::SubgroupMaxSize
                | BuiltIn::NumEnqueuedSubgroups
        )
    }

    fn is_ray_tracing(&self) -> bool {
        matches!(
            self,
            BuiltIn::LaunchIdKHR
                | BuiltIn::LaunchSizeKHR
                | BuiltIn::RayTminKHR
                | BuiltIn::RayTmaxKHR
                | BuiltIn::WorldRayOriginKHR
                | BuiltIn::WorldRayDirectionKHR
                | BuiltIn::ObjectRayOriginKHR
                | BuiltIn::ObjectRayDirectionKHR
                | BuiltIn::ObjectToWorldKHR
                | BuiltIn::WorldToObjectKHR
                | BuiltIn::InstanceCustomIndexKHR
                | BuiltIn::RayGeometryIndexKHR
                | BuiltIn::IncomingRayFlagsKHR
                | BuiltIn::CullMaskKHR
                | BuiltIn::HitKindKHR
                | BuiltIn::HitTNV
        )
    }

    fn requires_input_storage_class(&self) -> bool {
        // Most built-ins that are read-only inputs must be Input storage class
        matches!(
            self,
            // Fragment inputs
            BuiltIn::FragCoord
                | BuiltIn::PointCoord
                | BuiltIn::FrontFacing
                | BuiltIn::SampleId
                | BuiltIn::SamplePosition
                | BuiltIn::HelperInvocation
                | BuiltIn::FullyCoveredEXT
                // Barycentric coordinates
                | BuiltIn::BaryCoordKHR
                | BuiltIn::BaryCoordNoPerspKHR
                | BuiltIn::BaryCoordSmoothAMD
                | BuiltIn::BaryCoordSmoothCentroidAMD
                | BuiltIn::BaryCoordSmoothSampleAMD
                | BuiltIn::BaryCoordNoPerspAMD
                | BuiltIn::BaryCoordNoPerspCentroidAMD
                | BuiltIn::BaryCoordNoPerspSampleAMD
                | BuiltIn::BaryCoordPullModelAMD
                // Vertex/Instance index (vertex shader)
                | BuiltIn::VertexIndex
                | BuiltIn::VertexId
                | BuiltIn::InstanceIndex
                | BuiltIn::InstanceId
                | BuiltIn::BaseInstance
                | BuiltIn::BaseVertex
                | BuiltIn::DrawIndex
                // Compute inputs
                | BuiltIn::GlobalInvocationId
                | BuiltIn::LocalInvocationId
                | BuiltIn::LocalInvocationIndex
                | BuiltIn::NumWorkgroups
                | BuiltIn::WorkgroupId
                | BuiltIn::NumSubgroups
                | BuiltIn::SubgroupId
                | BuiltIn::SubgroupLocalInvocationId
                | BuiltIn::SubgroupSize
                // Subgroup masks
                | BuiltIn::SubgroupEqMask
                | BuiltIn::SubgroupGeMask
                | BuiltIn::SubgroupGtMask
                | BuiltIn::SubgroupLeMask
                | BuiltIn::SubgroupLtMask
                // Tessellation inputs
                | BuiltIn::TessCoord
                | BuiltIn::TessLevelOuter
                | BuiltIn::TessLevelInner
                | BuiltIn::PatchVertices
                // Geometry/Tessellation control inputs
                | BuiltIn::InvocationId
                // Ray tracing inputs
                | BuiltIn::LaunchIdKHR
                | BuiltIn::LaunchSizeKHR
                | BuiltIn::RayTminKHR
                | BuiltIn::RayTmaxKHR
                | BuiltIn::WorldRayOriginKHR
                | BuiltIn::WorldRayDirectionKHR
                | BuiltIn::ObjectRayOriginKHR
                | BuiltIn::ObjectRayDirectionKHR
                | BuiltIn::ObjectToWorldKHR
                | BuiltIn::WorldToObjectKHR
                | BuiltIn::InstanceCustomIndexKHR
                | BuiltIn::RayGeometryIndexKHR
                | BuiltIn::IncomingRayFlagsKHR
                | BuiltIn::CullMaskKHR
                | BuiltIn::HitKindKHR
                | BuiltIn::HitTNV
                // Shading rate (fragment input)
                | BuiltIn::ShadingRateKHR
                | BuiltIn::FragSizeEXT
                | BuiltIn::FragInvocationCountEXT
                // Device/View index
                | BuiltIn::DeviceIndex
                | BuiltIn::ViewIndex
                // SM built-ins
                | BuiltIn::WarpIDNV
                | BuiltIn::SMIDNV
                | BuiltIn::SMCountNV
                | BuiltIn::WarpsPerSMNV
                | BuiltIn::CoreIDARM
                | BuiltIn::CoreCountARM
                | BuiltIn::CoreMaxIDARM
                | BuiltIn::WarpIDARM
                | BuiltIn::WarpMaxIDARM
        )
    }

    fn requires_output_storage_class(&self) -> bool {
        matches!(
            self,
            // Outputs that must be written
            BuiltIn::FragDepth
                | BuiltIn::FragStencilRefEXT
                // Primitive outputs
                | BuiltIn::PrimitiveShadingRateKHR
                // Mesh shader outputs
                | BuiltIn::PrimitivePointIndicesEXT
                | BuiltIn::PrimitiveLineIndicesEXT
                | BuiltIn::PrimitiveTriangleIndicesEXT
                | BuiltIn::CullPrimitiveEXT
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_image_classification() {
        assert!(Op::ImageSampleImplicitLod.is_implicit_lod());
        assert!(Op::ImageSampleImplicitLod.is_sample());
        assert!(Op::ImageSampleImplicitLod.is_image_op());
        assert!(!Op::ImageSampleImplicitLod.is_explicit_lod());

        assert!(Op::ImageSampleExplicitLod.is_explicit_lod());
        assert!(Op::ImageSampleExplicitLod.is_sample());
        assert!(!Op::ImageSampleExplicitLod.is_implicit_lod());

        assert!(Op::ImageSampleDrefImplicitLod.is_dref());
        assert!(Op::ImageDrefGather.is_dref());
        assert!(Op::ImageDrefGather.is_gather());

        assert!(Op::ImageSampleProjImplicitLod.is_proj());
        assert!(Op::ImageFetch.is_fetch());
        assert!(Op::ImageRead.is_image_read_write());
        assert!(Op::ImageQuerySize.is_image_query());
    }

    #[test]
    fn test_op_atomic_classification() {
        assert!(Op::AtomicLoad.is_atomic());
        assert!(Op::AtomicStore.is_atomic());
        assert!(Op::AtomicIAdd.is_atomic());
        assert!(!Op::IAdd.is_atomic());
    }

    #[test]
    fn test_op_constant_classification() {
        assert!(Op::Constant.is_constant());
        assert!(Op::ConstantComposite.is_constant());
        assert!(Op::SpecConstant.is_spec_constant());
        assert!(Op::SpecConstant.is_scalar_spec_constant());
        assert!(Op::Undef.is_constant_or_undef());
        assert!(!Op::Undef.is_constant());
    }

    #[test]
    fn test_decoration_classification() {
        assert!(Decoration::Offset.is_member_only());
        assert!(Decoration::MatrixStride.is_member_only());
        assert!(!Decoration::Location.is_member_only());

        assert!(Decoration::Block.is_non_member());
        assert!(Decoration::Location.is_non_member());
        assert!(!Decoration::Offset.is_non_member());
    }

    #[test]
    fn test_builtin_classification() {
        assert!(BuiltIn::FragCoord.is_fragment_only());
        assert!(BuiltIn::BaryCoordKHR.is_fragment_only());
        assert!(BuiltIn::BaryCoordKHR.is_barycentric());

        assert!(BuiltIn::GlobalInvocationId.is_compute_only());
        assert!(BuiltIn::WorkDim.is_kernel_only());
        assert!(BuiltIn::CullPrimitiveEXT.is_mesh_output());
    }

    #[test]
    fn test_builtin_ray_tracing() {
        assert!(BuiltIn::LaunchIdKHR.is_ray_tracing());
        assert!(BuiltIn::HitKindKHR.is_ray_tracing());
        assert!(BuiltIn::RayTminKHR.is_ray_tracing());
        assert!(!BuiltIn::FragCoord.is_ray_tracing());
    }

    #[test]
    fn test_builtin_storage_class_requirements() {
        // Input-only built-ins
        assert!(BuiltIn::FragCoord.requires_input_storage_class());
        assert!(BuiltIn::VertexIndex.requires_input_storage_class());
        assert!(BuiltIn::GlobalInvocationId.requires_input_storage_class());
        assert!(BuiltIn::LaunchIdKHR.requires_input_storage_class());

        // Output-only built-ins
        assert!(BuiltIn::FragDepth.requires_output_storage_class());
        assert!(BuiltIn::PrimitivePointIndicesEXT.requires_output_storage_class());

        // Position and others can be both (not in either exclusive list)
        assert!(!BuiltIn::Position.requires_input_storage_class());
        assert!(!BuiltIn::Position.requires_output_storage_class());
    }

    #[test]
    fn test_op_type_classification() {
        // Scalar types
        assert!(Op::TypeInt.is_scalar_type());
        assert!(Op::TypeFloat.is_scalar_type());
        assert!(Op::TypeBool.is_scalar_type());
        assert!(Op::TypePointer.is_scalar_type());

        // Composite types
        assert!(Op::TypeVector.is_composite_type());
        assert!(Op::TypeMatrix.is_composite_type());
        assert!(Op::TypeArray.is_composite_type());
        assert!(Op::TypeStruct.is_composite_type());
        assert!(Op::TypeRuntimeArray.is_composite_type());
        assert!(Op::TypeCooperativeMatrixKHR.is_composite_type());

        // Cross-checks
        assert!(!Op::TypeVector.is_scalar_type());
        assert!(!Op::TypeInt.is_composite_type());
    }

    #[test]
    fn test_op_derivative_classification() {
        // All derivative operations
        assert!(Op::DPdx.is_derivative());
        assert!(Op::DPdy.is_derivative());
        assert!(Op::Fwidth.is_derivative());
        assert!(Op::DPdxFine.is_derivative());
        assert!(Op::DPdyFine.is_derivative());
        assert!(Op::FwidthFine.is_derivative());
        assert!(Op::DPdxCoarse.is_derivative());
        assert!(Op::DPdyCoarse.is_derivative());
        assert!(Op::FwidthCoarse.is_derivative());

        // Non-derivative operations
        assert!(!Op::FAdd.is_derivative());
        assert!(!Op::FSub.is_derivative());
        assert!(!Op::FMul.is_derivative());
    }

    #[test]
    fn test_op_barrier_classification() {
        // Barrier operations
        assert!(Op::ControlBarrier.is_barrier());
        assert!(Op::MemoryBarrier.is_barrier());
        assert!(Op::MemoryNamedBarrier.is_barrier());

        // Non-barrier operations
        assert!(!Op::NamedBarrierInitialize.is_barrier()); // Creates a barrier object, not a sync point
        assert!(!Op::Nop.is_barrier());
        assert!(!Op::Store.is_barrier());
    }

    #[test]
    fn test_op_terminator_classification() {
        // Terminator operations
        assert!(Op::Branch.is_terminator());
        assert!(Op::BranchConditional.is_terminator());
        assert!(Op::Switch.is_terminator());
        assert!(Op::Kill.is_terminator());
        assert!(Op::Return.is_terminator());
        assert!(Op::ReturnValue.is_terminator());
        assert!(Op::Unreachable.is_terminator());
        assert!(Op::TerminateInvocation.is_terminator());
        assert!(Op::IgnoreIntersectionKHR.is_terminator());
        assert!(Op::TerminateRayKHR.is_terminator());
        assert!(Op::EmitMeshTasksEXT.is_terminator());

        // Non-terminator operations
        assert!(!Op::Nop.is_terminator());
        assert!(!Op::FAdd.is_terminator());
        assert!(!Op::SelectionMerge.is_terminator());
    }

    #[test]
    fn test_op_merge_classification() {
        // Merge operations
        assert!(Op::SelectionMerge.is_merge());
        assert!(Op::LoopMerge.is_merge());

        // Non-merge operations
        assert!(!Op::Branch.is_merge());
        assert!(!Op::BranchConditional.is_merge());
        assert!(!Op::Nop.is_merge());
    }
}
