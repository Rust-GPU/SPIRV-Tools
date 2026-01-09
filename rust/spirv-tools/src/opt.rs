//! SPIR-V optimization support.

use crate::binary::Binary;
use crate::error::{Error, Message, MessageCallback, MessageLevel, TargetEnv};
use crate::val::ValidatorOptions;

/// Options for specifying the behavior of the optimizer.
#[derive(Default, Clone)]
pub struct Options {
    /// Records the validator options that should be passed to the validator,
    /// the validator will run with the options before optimizer.
    pub validator_options: Option<ValidatorOptions>,
    /// Records the maximum possible value for the id bound.
    pub max_id_bound: Option<u32>,
    /// Records whether all bindings within the module should be preserved.
    pub preserve_bindings: bool,
    /// Records whether all specialization constants within the module
    /// should be preserved.
    pub preserve_spec_constants: bool,
}

/// Available optimization passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Passes {
    AggressiveDCE,
    AmdExtToKhr,
    BlockMerge,
    ConditionalConstantPropagation,
    CFGCleanup,
    CodeSinking,
    CombineAccessChains,
    CompactIds,
    ConvertRelaxedToHalf,
    CopyPropagateArrays,
    DeadBranchElim,
    DeadInsertElim,
    DeadVariableElimination,
    DescriptorScalarReplacement,
    EliminateDeadConstant,
    EliminateDeadFunctions,
    EliminateDeadMembers,
    FixStorageClass,
    FlattenDecoration,
    FoldSpecConstantOpAndComposite,
    FreezeSpecConstantValue,
    GraphicsRobustAccess,
    IfConversion,
    InlineExhaustive,
    InlineOpaque,
    InsertExtractElim,
    InterpolateFixup,
    LocalAccessChainConvert,
    LocalMultiStoreElim,
    LocalRedundancyElimination,
    LocalSingleBlockLoadStoreElim,
    LocalSingleStoreElim,
    LoopInvariantCodeMotion,
    LoopPeeling,
    LoopUnswitch,
    MergeReturn,
    Null,
    PrivateToLocal,
    PropagateLineInfo,
    ReduceLoadSize,
    RedundancyElimination,
    RedundantLineInfoElim,
    RelaxFloatOps,
    RemoveDuplicates,
    RemoveUnusedInterfaceVariables,
    ReplaceInvalidOpcode,
    Simplification,
    SSARewrite,
    StrengthReduction,
    StripDebugInfo,
    StripNonSemanticInfo,
    UnifyConstant,
    UpgradeMemoryModel,
    VectorDCE,
    Workaround1209,
    WrapOpKill,
}

/// Trait for SPIR-V optimizers.
pub trait Optimizer {
    fn with_env(target_env: TargetEnv) -> Self;

    fn optimize<MC: MessageCallback>(
        &self,
        input: impl AsRef<[u32]>,
        msg_callback: &mut MC,
        options: Option<Options>,
    ) -> Result<Binary, Error>;

    /// Register a single pass with the optimizer.
    fn register_pass(&mut self, pass: Passes) -> &mut Self;

    /// Registers passes that attempt to improve performance of generated code.
    fn register_performance_passes(&mut self) -> &mut Self;

    /// Registers passes that attempt to improve the size of generated code.
    fn register_size_passes(&mut self) -> &mut Self;

    /// Registers passes that attempt to legalize the generated code.
    fn register_hlsl_legalization_passes(&mut self) -> &mut Self;
}

/// Create an optimizer for the given target environment.
pub fn create(te: Option<TargetEnv>) -> impl Optimizer {
    let target_env = te.unwrap_or_default();
    RustOptimizer::with_env(target_env)
}

/// A pure Rust implementation of the SPIR-V optimizer.
#[allow(dead_code)]
pub struct RustOptimizer {
    target_env: TargetEnv,
    passes: Vec<Passes>,
}

impl Default for RustOptimizer {
    fn default() -> Self {
        Self {
            target_env: TargetEnv::default(),
            passes: Vec::new(),
        }
    }
}

impl Optimizer for RustOptimizer {
    fn with_env(target_env: TargetEnv) -> Self {
        Self {
            target_env,
            passes: Vec::new(),
        }
    }

    fn optimize<MC: MessageCallback>(
        &self,
        input: impl AsRef<[u32]>,
        msg_callback: &mut MC,
        _options: Option<Options>,
    ) -> Result<Binary, Error> {
        let words = input.as_ref();

        // If no passes registered, return input unchanged
        if self.passes.is_empty() {
            return Ok(Binary::OwnedU32(words.to_vec()));
        }

        // Use the e-graph based optimizer which applies all optimizations
        // in a single global pass using equality saturation
        match spirv_tools_opt::optimize_words(words) {
            Ok(optimized) => Ok(Binary::OwnedU32(optimized)),
            Err(e) => {
                msg_callback.on_message(Message {
                    level: MessageLevel::Error,
                    source: None,
                    line: 0,
                    column: 0,
                    index: 0,
                    message: format!("Optimization failed: {e}"),
                    notes: String::new(),
                });
                // On error, return input unchanged
                Ok(Binary::OwnedU32(words.to_vec()))
            }
        }
    }

    fn register_pass(&mut self, pass: Passes) -> &mut Self {
        self.passes.push(pass);
        self
    }

    fn register_performance_passes(&mut self) -> &mut Self {
        // These are the passes that spirv-opt uses for -O
        self.passes.extend([
            Passes::InlineExhaustive,
            Passes::LocalAccessChainConvert,
            Passes::LocalSingleBlockLoadStoreElim,
            Passes::LocalSingleStoreElim,
            Passes::DeadBranchElim,
            Passes::LocalMultiStoreElim,
            Passes::AggressiveDCE,
            Passes::BlockMerge,
            Passes::InsertExtractElim,
            Passes::RedundancyElimination,
            Passes::ConditionalConstantPropagation,
            Passes::Simplification,
            Passes::CFGCleanup,
            Passes::DeadVariableElimination,
            Passes::EliminateDeadFunctions,
        ]);
        self
    }

    fn register_size_passes(&mut self) -> &mut Self {
        // These are the passes that spirv-opt uses for -Os
        self.passes.extend([
            Passes::InlineExhaustive,
            Passes::LocalAccessChainConvert,
            Passes::LocalSingleBlockLoadStoreElim,
            Passes::LocalSingleStoreElim,
            Passes::DeadBranchElim,
            Passes::LocalMultiStoreElim,
            Passes::AggressiveDCE,
            Passes::BlockMerge,
            Passes::RedundancyElimination,
            Passes::CFGCleanup,
            Passes::DeadVariableElimination,
            Passes::EliminateDeadFunctions,
            Passes::EliminateDeadConstant,
        ]);
        self
    }

    fn register_hlsl_legalization_passes(&mut self) -> &mut Self {
        self.passes.extend([
            Passes::InlineExhaustive,
            Passes::DeadBranchElim,
            Passes::MergeReturn,
            Passes::InlineOpaque,
            Passes::LocalAccessChainConvert,
            Passes::LocalSingleBlockLoadStoreElim,
            Passes::LocalSingleStoreElim,
            Passes::DeadBranchElim,
            Passes::LocalMultiStoreElim,
            Passes::AggressiveDCE,
            Passes::CopyPropagateArrays,
            Passes::BlockMerge,
            Passes::RedundancyElimination,
            Passes::DeadBranchElim,
            Passes::LocalMultiStoreElim,
            Passes::BlockMerge,
            Passes::Simplification,
            Passes::CFGCleanup,
            Passes::DeadVariableElimination,
            Passes::EliminateDeadFunctions,
        ]);
        self
    }
}
