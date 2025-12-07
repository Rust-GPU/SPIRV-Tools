use std::marker::PhantomData;

use arbitrary::{Arbitrary, Unstructured};
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder, InsertPoint, Instruction, Module};
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel, Op};
use spirv_tools_core::validation::validate_module;
use spirv_tools_core::TargetEnv;

/// Marker types to track validity at the type level.
pub enum Valid {}
pub enum Unchecked {}
pub enum IntentionallyInvalid {}

#[derive(Debug, Clone, Copy, Arbitrary)]
pub enum InvalidKind {
    MissingMemoryModel,
    MissingTerminator,
}

pub struct FuzzConfig {
    pub seed: u64,
    pub prefer_valid: bool,
    pub allow_invalid: bool,
}

/// Wrapper around an rspirv module with a validity marker.
pub struct FuzzModule<V> {
    module: Module,
    _marker: PhantomData<V>,
}

impl<V> FuzzModule<V> {
    pub fn into_words(self) -> Vec<u32> {
        self.module.assemble()
    }
}

impl Arbitrary<'_> for FuzzModule<Unchecked> {
    fn arbitrary(u: &mut Unstructured<'_>) -> arbitrary::Result<Self> {
        let mut builder = Builder::new();
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = builder.type_void();
        let func_ty = builder.type_function(void, vec![]);

        let func_count = u.int_in_range::<u32>(1..=2)?;
        for _ in 0..func_count {
            builder
                .begin_function(void, None, FunctionControl::NONE, func_ty)
                .map_err(|_| arbitrary::Error::IncorrectFormat)?;
            builder
                .begin_block(None)
                .map_err(|_| arbitrary::Error::IncorrectFormat)?;

            let nop_count = u.int_in_range::<u32>(0..=2)?;
            for _ in 0..nop_count {
                let _ = builder.insert_into_block(
                    InsertPoint::End,
                    Instruction::new(Op::Nop, None, None, Vec::new()),
                );
            }

            builder.ret().map_err(|_| arbitrary::Error::IncorrectFormat)?;
            builder
                .end_function()
                .map_err(|_| arbitrary::Error::IncorrectFormat)?;
        }

        let module = builder.module();
        Ok(FuzzModule {
            module,
            _marker: PhantomData,
        })
    }

    fn arbitrary_take_rest(mut u: Unstructured<'_>) -> arbitrary::Result<Self> {
        Self::arbitrary(&mut u)
    }
}

impl FuzzModule<Unchecked> {
    pub fn into_valid(self, env: TargetEnv) -> Result<FuzzModule<Valid>, String> {
        validate_module(&self.module.assemble(), env).map_err(|err| err.to_string())?;
        Ok(FuzzModule {
            module: self.module,
            _marker: PhantomData,
        })
    }

    pub fn into_invalid(mut self, kind: InvalidKind) -> FuzzModule<IntentionallyInvalid> {
        apply_invalid_mutation(&mut self.module, kind);
        FuzzModule {
            module: self.module,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FuzzOutcome {
    Valid { words: Vec<u32> },
    Invalid { words: Vec<u32>, kind: InvalidKind },
}

pub struct FuzzGenerator {
    cfg: FuzzConfig,
}

impl FuzzGenerator {
    pub fn new(cfg: FuzzConfig) -> Self {
        Self { cfg }
    }

    pub fn generate(&self, env: TargetEnv, input: &[u8]) -> Result<FuzzOutcome, String> {
        let mut data = Vec::new();
        let mut rng = fastrand::Rng::with_seed(self.cfg.seed);
        let extra_len = 64usize;
        for _ in 0..extra_len {
            data.push(rng.u8(..));
        }
        data.extend_from_slice(input);
        let mut u = Unstructured::new(&data);
        let unchecked: FuzzModule<Unchecked> = Arbitrary::arbitrary(&mut u)
            .map_err(|e| format!("fuzz gen failed: {e:?}"))?;

        if self.cfg.prefer_valid || !self.cfg.allow_invalid {
            let valid = unchecked.into_valid(env)?;
            return Ok(FuzzOutcome::Valid {
                words: valid.into_words(),
            });
        }

        let invalid_kind: InvalidKind = Arbitrary::arbitrary(&mut u)
            .unwrap_or(InvalidKind::MissingMemoryModel);
        let invalid = unchecked.into_invalid(invalid_kind);
        Ok(FuzzOutcome::Invalid {
            words: invalid.into_words(),
            kind: invalid_kind,
        })
    }
}

fn apply_invalid_mutation(module: &mut dr::Module, kind: InvalidKind) {
    match kind {
        InvalidKind::MissingMemoryModel => {
            module.memory_model = None;
        }
        InvalidKind::MissingTerminator => {
            if let Some(func) = module.functions.first_mut() {
                if let Some(block) = func.blocks.first_mut() {
                    if !block.instructions.is_empty() {
                        block.instructions.pop();
                    }
                }
            }
        }
    }
}
