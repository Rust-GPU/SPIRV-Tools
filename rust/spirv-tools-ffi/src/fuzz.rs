use std::marker::PhantomData;

use arbitrary::{Arbitrary, Unstructured};
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder, InsertPoint, Instruction, Module, Operand};
use rspirv::spirv::{
    self, AddressingModel, Capability, ExecutionModel, FunctionControl, LoopControl, MemoryModel,
    Op, SelectionControl, StorageClass,
};
use spirv_tools_core::validation::validate_module;
use spirv_tools_core::TargetEnv;

/// Marker types to track validity at the type level.
pub enum Valid {}
pub enum Unchecked {}
pub enum IntentionallyInvalid {}

#[derive(Debug, Clone, Copy, Arbitrary, PartialEq, Eq)]
pub enum InvalidKind {
    MissingMemoryModel,
    MissingTerminator,
    MissingEntryPoint,
    TypeMismatch,
    BrokenIdBound,
    DanglingUse,
    DuplicateId,
    MissingSelectionMerge,
    PhiPredecessorMismatch,
    StorageClassMismatch,
    MissingLoopMerge,
    AccessChainOvershoot,
    InvalidDecorationTarget,
    DuplicateBinding,
    DuplicateEntryPointInterface,
    RayPayloadInterfaceOnNonRayEntry,
    CallableDataInterfaceOnNonRayEntry,
    HitAttributeInterfaceOnNonRayEntry,
    DuplicateRayPayloadInterface,
    DuplicateCallableDataInterface,
    DuplicateHitAttributeInterface,
    RayEntryWithNonRayInterface,
    MixedRayInterfaceStorageClasses,
    MissingRayExecutionModel,
    MissingRayCapability,
    RayPayloadTypeMismatch,
    CallableDataTypeMismatch,
    HitAttributeTypeMismatch,
    HitAttributeOnRayGen,
    RayEntryWithWorkgroupInterface,
    RayEntryWithOutputInterface,
    RayEntryWithMixedIoInterfaces,
    RayEntryWithPrivateInterface,
    RayEntryWithFunctionInterface,
    RayEntryWithCrossWorkgroupInterface,
    RayEntryWithGenericInterface,
    RayEntryWithUniformConstantInterface,
    RayEntryWithPushConstantInterface,
    RayEntryWithUniformInterface,
    RayEntryWithStorageBufferInterface,
    RayEntryWithShaderRecordBufferInterface,
    RayEntryWithTaskPayloadInterface,
    RayEntryWithAtomicCounterInterface,
    RayEntryWithImageInterface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validity {
    Valid,
    Invalid(InvalidKind),
    Unchecked,
}

impl<'a> Arbitrary<'a> for Validity {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        if u.ratio(1, 4)? {
            let kind = InvalidKind::arbitrary(u)?;
            return Ok(Validity::Invalid(kind));
        }
        if u.ratio(1, 4)? {
            return Ok(Validity::Unchecked);
        }
        Ok(Validity::Valid)
    }

    fn arbitrary_take_rest(mut u: Unstructured<'a>) -> arbitrary::Result<Self> {
        Self::arbitrary(&mut u)
    }
}

#[derive(Debug, Clone)]
pub struct MaybeInvalid<T> {
    value: T,
    validity: Validity,
}

impl<T> MaybeInvalid<T> {
    pub fn new(value: T, validity: Validity) -> Self {
        Self { value, validity }
    }

    pub fn validity(&self) -> Validity {
        self.validity
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> MaybeInvalid<U> {
        MaybeInvalid {
            value: f(self.value),
            validity: self.validity,
        }
    }

    pub fn into_inner(self) -> T {
        self.value
    }

    pub fn with_validity(self, validity: Validity) -> Self {
        MaybeInvalid {
            value: self.value,
            validity,
        }
    }
}

pub struct FuzzConfig {
    pub seed: u64,
    pub prefer_valid: bool,
    pub allow_invalid: bool,
    pub invalid_hint: Option<InvalidKind>,
}

/// Wrapper around an rspirv module with a validity marker.
pub struct FuzzModule<V> {
    module: Module,
    _marker: PhantomData<V>,
}

impl<V> Clone for FuzzModule<V> {
    fn clone(&self) -> Self {
        FuzzModule {
            module: self.module.clone(),
            _marker: PhantomData,
        }
    }
}

impl<V> FuzzModule<V> {
    pub fn into_words(self) -> Vec<u32> {
        self.module.assemble()
    }

    pub fn module(&self) -> &Module {
        &self.module
    }
}

impl Arbitrary<'_> for FuzzModule<Unchecked> {
    fn arbitrary(u: &mut Unstructured<'_>) -> arbitrary::Result<Self> {
        let result: arbitrary::Result<FuzzModule<Unchecked>> = (|| {
            let mut builder = Builder::new();
            builder.capability(Capability::Shader);
            builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

            let void = builder.type_void();
            let func_ty = builder.type_function(void, vec![]);
            let int_ty = builder.type_int(32, 0);
            let bool_ty = builder.type_bool();
            let array_len = builder.constant_bit32(int_ty, 4);
            let array_int = builder.type_array(int_ty, array_len);
            let struct_ty = builder.type_struct(vec![int_ty, array_int]);
            let ptr_fn_int = builder.type_pointer(None, StorageClass::Function, int_ty);
            let ptr_fn_struct = builder.type_pointer(None, StorageClass::Function, struct_ty);
            let ptr_fn_array = builder.type_pointer(None, StorageClass::Function, array_int);
            let zero_const = builder.constant_bit32(int_ty, 0);
            let one_const = builder.constant_bit32(int_ty, 1);
            let cond_true = builder.constant_true(bool_ty);
            let cond_false = builder.constant_false(bool_ty);
            let array_const = builder.constant_composite(
                array_int,
                vec![zero_const, zero_const, zero_const, zero_const],
            );
            let struct_const = builder.constant_composite(struct_ty, vec![zero_const, array_const]);
            let struct_ptr_member = builder.id();
            let _ = builder.insert_into_block(
                InsertPoint::End,
                Instruction::new(
                    Op::CompositeExtract,
                    Some(int_ty),
                    Some(struct_ptr_member),
                    vec![struct_const.into(), 0u32.into()],
                ),
            );
            let struct_rebuilt = builder.id();
            let _ = builder.insert_into_block(
                InsertPoint::End,
                Instruction::new(
                    Op::CompositeInsert,
                    Some(struct_ty),
                    Some(struct_rebuilt),
                    vec![struct_ptr_member.into(), struct_const.into(), 0u32.into()],
                ),
            );

            let func_count = u.int_in_range::<u32>(1..=2)?;
            let mut first_func_id = None;
            for _ in 0..func_count {
                let func_id = builder
                    .begin_function(void, None, FunctionControl::NONE, func_ty)
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                if first_func_id.is_none() {
                    first_func_id = Some(func_id);
                }
                builder
                    .begin_block(None)
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?;

                let nop_count = u.int_in_range::<u32>(1..=3)?;
                for _ in 0..nop_count {
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(Op::Nop, None, None, Vec::new()),
                    );
                }

                // Optionally exercise a store in a function-scoped variable to get
                // some type/global coverage.
                if u.ratio(1, 2)? {
                    let var_id = builder.variable(ptr_fn_int, None, StorageClass::Function, None);
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![var_id.into(), zero_const.into()],
                        ),
                    );
                }

                // Optionally store a composite struct into a function-scoped variable
                // to exercise aggregate types.
                if u.ratio(1, 3)? {
                    let struct_var =
                        builder.variable(ptr_fn_struct, None, StorageClass::Function, None);
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![struct_var.into(), struct_rebuilt.into()],
                        ),
                    );

                    // Access an array element inside the struct and load it.
                    let access_chain_id = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::AccessChain,
                            Some(ptr_fn_int),
                            Some(access_chain_id),
                            vec![struct_var.into(), 1u32.into(), 0u32.into()],
                        ),
                    );
                    let load_struct_elem = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Load,
                            Some(int_ty),
                            Some(load_struct_elem),
                            vec![access_chain_id.into()],
                        ),
                    );

                    // Also create an access into the array directly for mutation hooks.
                    let array_var =
                        builder.variable(ptr_fn_array, None, StorageClass::Function, None);
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![array_var.into(), array_const.into()],
                        ),
                    );
                    let array_access = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::AccessChain,
                            Some(ptr_fn_int),
                            Some(array_access),
                            vec![array_var.into(), 2u32.into()],
                        ),
                    );
                    let load_array_elem = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Load,
                            Some(int_ty),
                            Some(load_array_elem),
                            vec![array_access.into()],
                        ),
                    );
                }

                if u.ratio(1, 2)? {
                    let fresh = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::IAdd,
                            Some(int_ty),
                            Some(fresh),
                            vec![zero_const.into(), one_const.into()],
                        ),
                    );
                }

                // Optionally add a loop with merge/continue and a conditional exit.
                if u.ratio(1, 3)? {
                    let merge_label = builder.id();
                    let continue_label = builder.id();
                    let body_label = builder.id();
                    let loop_cond = if u.ratio(1, 2)? {
                        cond_true
                    } else {
                        cond_false
                    };

                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![
                                merge_label.into(),
                                continue_label.into(),
                                LoopControl::NONE.bits().into(),
                            ],
                        ),
                    );
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(Op::Branch, None, None, vec![continue_label.into()]),
                    );

                    builder
                        .begin_block(Some(continue_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(Op::Branch, None, None, vec![body_label.into()]),
                    );

                    builder
                        .begin_block(Some(body_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![loop_cond.into(), continue_label.into(), merge_label.into()],
                        ),
                    );

                    builder
                        .begin_block(Some(merge_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                }

                // Optionally add a simple if-else with a phi in the merge block to
                // exercise control flow and SSA edges.
                if u.ratio(1, 2)? {
                    let merge_label = builder.id();
                    let then_label = builder.id();
                    let else_label = builder.id();
                    let cond = if u.ratio(1, 2)? {
                        cond_true
                    } else {
                        cond_false
                    };

                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::SelectionMerge,
                            None,
                            None,
                            vec![merge_label.into(), SelectionControl::NONE.bits().into()],
                        ),
                    );
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![cond.into(), then_label.into(), else_label.into()],
                        ),
                    );

                    builder
                        .begin_block(Some(then_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(Op::Branch, None, None, vec![merge_label.into()]),
                    );

                    builder
                        .begin_block(Some(else_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(Op::Branch, None, None, vec![merge_label.into()]),
                    );

                    builder
                        .begin_block(Some(merge_label))
                        .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                    let phi_id = builder.id();
                    let _ = builder.insert_into_block(
                        InsertPoint::End,
                        Instruction::new(
                            Op::Phi,
                            Some(int_ty),
                            Some(phi_id),
                            vec![
                                zero_const.into(),
                                then_label.into(),
                                one_const.into(),
                                else_label.into(),
                            ],
                        ),
                    );
                }

                builder
                    .ret()
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?;
                builder
                    .end_function()
                    .map_err(|_| arbitrary::Error::IncorrectFormat)?;
            }

            if let Some(func_id) = first_func_id {
                builder.entry_point(ExecutionModel::Vertex, func_id, "main", Vec::new());
            }

            let mut module = builder.module();
            ensure_nop_present(&mut module);
            Ok(FuzzModule {
                module,
                _marker: PhantomData,
            })
        })();

        result
    }

    fn arbitrary_take_rest(mut u: Unstructured<'_>) -> arbitrary::Result<Self> {
        Self::arbitrary(&mut u)
    }
}

impl Arbitrary<'_> for MaybeInvalid<FuzzModule<Unchecked>> {
    fn arbitrary(u: &mut Unstructured<'_>) -> arbitrary::Result<Self> {
        let module = FuzzModule::<Unchecked>::arbitrary(u)?;
        let validity = Validity::arbitrary(u).unwrap_or(Validity::Unchecked);
        Ok(MaybeInvalid {
            value: module,
            validity,
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
        let candidate: MaybeInvalid<FuzzModule<Unchecked>> = Arbitrary::arbitrary(&mut u)
            .unwrap_or_else(|_| {
                let module = minimal_valid_module_with_tag(self.cfg.seed as u32);
                MaybeInvalid::new(
                    FuzzModule {
                        module,
                        _marker: PhantomData,
                    },
                    Validity::Valid,
                )
            });

        self.materialize(env, candidate, &mut u)
    }

    fn materialize(
        &self,
        env: TargetEnv,
        candidate: MaybeInvalid<FuzzModule<Unchecked>>,
        u: &mut Unstructured<'_>,
    ) -> Result<FuzzOutcome, String> {
        let candidate_clone = candidate.clone();
        let want_invalid = self.cfg.allow_invalid
            && !self.cfg.prefer_valid
            && (self.cfg.invalid_hint.is_some()
                || matches!(candidate.validity(), Validity::Invalid(_)));

        if want_invalid {
            let invalid_kind = self.choose_invalid_kind(candidate.validity(), u);
            let invalid = candidate.value.into_invalid(invalid_kind);
            return Ok(FuzzOutcome::Invalid {
                words: invalid.into_words(),
                kind: invalid_kind,
            });
        }

        let valid = match candidate.value.into_valid(env) {
            Ok(v) => v,
            Err(_) if self.cfg.allow_invalid => {
                let fallback_kind =
                    self.choose_invalid_kind(Validity::Invalid(InvalidKind::TypeMismatch), u);
                return Ok(FuzzOutcome::Invalid {
                    words: candidate_clone.into_inner().into_words(),
                    kind: fallback_kind,
                });
            }
            Err(_) => {
                return Ok(FuzzOutcome::Valid {
                    words: minimal_valid_words(),
                })
            }
        };
        Ok(FuzzOutcome::Valid {
            words: valid.into_words(),
        })
    }

    fn choose_invalid_kind(&self, validity: Validity, u: &mut Unstructured<'_>) -> InvalidKind {
        if let Some(kind) = self.cfg.invalid_hint {
            return kind;
        }
        if let Validity::Invalid(kind) = validity {
            return kind;
        }
        Arbitrary::arbitrary(u).unwrap_or(InvalidKind::MissingMemoryModel)
    }
}

impl FuzzGenerator {
    #[doc(hidden)]
    pub fn materialize_for_test(
        &self,
        env: TargetEnv,
        candidate: MaybeInvalid<FuzzModule<Unchecked>>,
    ) -> Result<FuzzOutcome, String> {
        let mut empty = Unstructured::new(&[]);
        self.materialize(env, candidate, &mut empty)
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
        InvalidKind::MissingEntryPoint => {
            module.entry_points.clear();
        }
        InvalidKind::TypeMismatch => {
            if let Some(func) = module.functions.first_mut() {
                if let Some(block) = func.blocks.first_mut() {
                    if let Some(inst) = block.instructions.first_mut() {
                        inst.result_type = Some(0);
                    }
                }
            }
        }
        InvalidKind::BrokenIdBound => {
            if let Some(header) = module.header.as_mut() {
                header.bound = 1;
            } else {
                module.header = Some(dr::ModuleHeader {
                    magic_number: spirv::MAGIC_NUMBER,
                    version: ((spirv::MAJOR_VERSION as u32) << 16)
                        | ((spirv::MINOR_VERSION as u32) << 8),
                    generator: 0,
                    bound: 1,
                    reserved_word: 0,
                });
            }
        }
        InvalidKind::DanglingUse => {
            module.types_global_values.push(Instruction::new(
                Op::IAdd,
                Some(99),
                Some(98),
                vec![97u32.into(), 96u32.into()],
            ));
        }
        InvalidKind::DuplicateId => {
            let int_ty = module
                .types_global_values
                .iter()
                .find_map(|inst| {
                    if inst.class.opcode == Op::TypeInt {
                        inst.result_id
                    } else {
                        None
                    }
                })
                .unwrap_or(2);
            module.types_global_values.push(Instruction::new(
                Op::Constant,
                Some(int_ty),
                Some(50),
                vec![0u32.into()],
            ));
            module.types_global_values.push(Instruction::new(
                Op::Constant,
                Some(int_ty),
                Some(50),
                vec![1u32.into()],
            ));
        }
        InvalidKind::MissingSelectionMerge => {
            let bool_ty = module
                .types_global_values
                .iter()
                .find_map(|inst| (inst.class.opcode == Op::TypeBool).then_some(inst.result_id))
                .flatten()
                .unwrap_or_else(|| {
                    let id = fresh_id(module);
                    module.types_global_values.push(Instruction::new(
                        Op::TypeBool,
                        None,
                        Some(id),
                        vec![],
                    ));
                    id
                });
            let cond_id = module
                .types_global_values
                .iter()
                .find_map(|inst| {
                    (inst.class.opcode == Op::ConstantTrue && inst.result_type == Some(bool_ty))
                        .then_some(inst.result_id)
                })
                .flatten()
                .unwrap_or_else(|| {
                    let id = fresh_id(module);
                    module.types_global_values.push(Instruction::new(
                        Op::ConstantTrue,
                        Some(bool_ty),
                        Some(id),
                        vec![],
                    ));
                    id
                });
            let fallback_label_id = fresh_id(module);
            if let Some(func) = module.functions.first_mut() {
                if let Some(block) = func.blocks.first_mut() {
                    let before = block.instructions.len();
                    block
                        .instructions
                        .retain(|inst| inst.class.opcode != Op::SelectionMerge);
                    if before == block.instructions.len() {
                        let label_id = block
                            .label
                            .as_ref()
                            .and_then(|inst| inst.result_id)
                            .unwrap_or(fallback_label_id);
                        if block.label.is_none() {
                            block.label = Some(Instruction::new(
                                Op::Label,
                                None,
                                Some(label_id),
                                Vec::new(),
                            ));
                        }
                        block.instructions.push(Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![cond_id.into(), label_id.into(), label_id.into()],
                        ));
                    }
                }
            }
        }
        InvalidKind::PhiPredecessorMismatch => {
            let mut touched = false;
            for func in &mut module.functions {
                for block in &mut func.blocks {
                    if let Some(phi) = block
                        .instructions
                        .iter_mut()
                        .find(|inst| inst.class.opcode == Op::Phi)
                    {
                        if phi.operands.len() >= 2 {
                            phi.operands.truncate(phi.operands.len() - 2);
                        }
                        touched = true;
                        break;
                    }
                }
            }
            if !touched {
                let int_ty = module
                    .types_global_values
                    .iter()
                    .find_map(|inst| (inst.class.opcode == Op::TypeInt).then_some(inst.result_id))
                    .flatten()
                    .unwrap_or_else(|| {
                        let id = fresh_id(module);
                        module.types_global_values.push(Instruction::new(
                            Op::TypeInt,
                            None,
                            Some(id),
                            vec![32u32.into(), 0u32.into()],
                        ));
                        id
                    });
                let phi_id = fresh_id(module);
                if let Some(func) = module.functions.first_mut() {
                    if let Some(block) = func.blocks.first_mut() {
                        block.instructions.push(Instruction::new(
                            Op::Phi,
                            Some(int_ty),
                            Some(phi_id),
                            Vec::new(),
                        ));
                    }
                }
            }
        }
        InvalidKind::StorageClassMismatch => {
            if !mutate_variable_storage_class(module) {
                insert_mismatched_variable(module);
            }
        }
        InvalidKind::MissingLoopMerge => {
            let mut removed = false;
            for func in &mut module.functions {
                for block in &mut func.blocks {
                    let before = block.instructions.len();
                    block
                        .instructions
                        .retain(|inst| inst.class.opcode != Op::LoopMerge);
                    if before != block.instructions.len() {
                        removed = true;
                    }
                }
            }
            if !removed {
                let header = ensure_header(module);
                header.bound = 1;
            }
        }
        InvalidKind::AccessChainOvershoot => {
            if !widen_access_chain(module) {
                inject_bad_access_chain(module);
            }
        }
        InvalidKind::InvalidDecorationTarget => {
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    999u32.into(),
                    spirv::Decoration::Binding.into(),
                    0u32.into(),
                ],
            ));
        }
        InvalidKind::DuplicateBinding => {
            let (ptr_ty, vars) = ensure_uniform_vars(module, 2);
            for id in &vars {
                module.annotations.push(Instruction::new(
                    Op::Decorate,
                    None,
                    None,
                    vec![(*id).into(), spirv::Decoration::Binding.into(), 0u32.into()],
                ));
            }
            for var_id in vars {
                if !module
                    .types_global_values
                    .iter()
                    .any(|inst| inst.result_id == Some(var_id))
                {
                    module.types_global_values.insert(
                        0,
                        Instruction::new(
                            Op::Variable,
                            Some(ptr_ty),
                            Some(var_id),
                            vec![StorageClass::UniformConstant.into()],
                        ),
                    );
                }
            }
        }
        InvalidKind::DuplicateEntryPointInterface => {
            if let Some(ep) = module.entry_points.first_mut() {
                if let Some(first) = ep.operands.get(2).cloned() {
                    ep.operands.push(first);
                } else {
                    ep.operands.push(1u32.into());
                    ep.operands.push(1u32.into());
                }
            }
        }
        InvalidKind::RayPayloadInterfaceOnNonRayEntry => {
            if module.entry_points.is_empty() || module.functions.is_empty() {
                *module = minimal_valid_module_with_tag(0);
            }
            let int_ty = ensure_int_type(module);
            let payload_ptr =
                ensure_pointer_type(module, StorageClass::IncomingRayPayloadKHR, int_ty);
            let var_id = fresh_id(module);
            module.types_global_values.push(Instruction::new(
                Op::Variable,
                Some(payload_ptr),
                Some(var_id),
                vec![StorageClass::IncomingRayPayloadKHR.into()],
            ));
            if let Some(ep) = module.entry_points.first_mut() {
                // Force a non-ray execution model so the interface storage class is invalid.
                if let Some(model) = ep.operands.get_mut(0) {
                    *model = ExecutionModel::Vertex.into();
                }
                ep.operands.push(var_id.into());
            }
        }
        InvalidKind::CallableDataInterfaceOnNonRayEntry => {
            if module.entry_points.is_empty() || module.functions.is_empty() {
                *module = minimal_valid_module_with_tag(0);
            }
            let int_ty = ensure_int_type(module);
            let ptr =
                ensure_pointer_type(module, StorageClass::IncomingCallableDataKHR, int_ty);
            let var_id = fresh_id(module);
            module.types_global_values.push(Instruction::new(
                Op::Variable,
                Some(ptr),
                Some(var_id),
                vec![StorageClass::IncomingCallableDataKHR.into()],
            ));
            if let Some(ep) = module.entry_points.first_mut() {
                if let Some(model) = ep.operands.get_mut(0) {
                    *model = ExecutionModel::Fragment.into();
                }
                ep.operands.push(var_id.into());
            }
        }
        InvalidKind::HitAttributeInterfaceOnNonRayEntry => {
            if module.entry_points.is_empty() || module.functions.is_empty() {
                *module = minimal_valid_module_with_tag(0);
            }
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::HitAttributeKHR, int_ty);
            let var_id = fresh_id(module);
            module.types_global_values.push(Instruction::new(
                Op::Variable,
                Some(ptr),
                Some(var_id),
                vec![StorageClass::HitAttributeKHR.into()],
            ));
            if let Some(ep) = module.entry_points.first_mut() {
                if let Some(model) = ep.operands.get_mut(0) {
                    *model = ExecutionModel::GLCompute.into();
                }
                ep.operands.push(var_id.into());
            }
        }
        InvalidKind::DuplicateRayPayloadInterface => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::IncomingRayPayloadKHR, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::IncomingRayPayloadKHR, 2);
            push_interfaces(module, &ids);
        }
        InvalidKind::DuplicateCallableDataInterface => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::IncomingCallableDataKHR, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::IncomingCallableDataKHR, 2);
            push_interfaces(module, &ids);
        }
        InvalidKind::DuplicateHitAttributeInterface => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::HitAttributeKHR, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::HitAttributeKHR, 2);
            push_interfaces(module, &ids);
        }
        InvalidKind::RayEntryWithNonRayInterface => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::Input, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::Input, 1);
            push_interfaces(module, &ids);
        }
        InvalidKind::MixedRayInterfaceStorageClasses => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let input_ptr = ensure_pointer_type(module, StorageClass::Input, int_ty);
            let uniform_ptr =
                ensure_pointer_type(module, StorageClass::UniformConstant, int_ty);
            let ids_input = insert_global_vars(module, input_ptr, StorageClass::Input, 1);
            let ids_uniform =
                insert_global_vars(module, uniform_ptr, StorageClass::UniformConstant, 1);
            let mut ids = ids_input;
            ids.extend(ids_uniform);
            push_interfaces(module, &ids);
        }
        InvalidKind::RayEntryWithWorkgroupInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Workgroup);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithOutputInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Output);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithMixedIoInterfaces => {
            ensure_ray_entry_point(module);
            let input = insert_interface_var(module, StorageClass::Input);
            let output = insert_interface_var(module, StorageClass::Output);
            push_interfaces(module, &[input, output]);
        }
        InvalidKind::RayEntryWithPrivateInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Private);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithFunctionInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Function);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithCrossWorkgroupInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::CrossWorkgroup);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithGenericInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Generic);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithUniformConstantInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::UniformConstant);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithPushConstantInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::PushConstant);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithUniformInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Uniform);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithStorageBufferInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::StorageBuffer);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithShaderRecordBufferInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::ShaderRecordBufferKHR);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithTaskPayloadInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::TaskPayloadWorkgroupEXT);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithAtomicCounterInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::AtomicCounter);
            push_interfaces(module, &[id]);
        }
        InvalidKind::RayEntryWithImageInterface => {
            ensure_ray_entry_point(module);
            let id = insert_interface_var(module, StorageClass::Image);
            push_interfaces(module, &[id]);
        }
        InvalidKind::MissingRayExecutionModel => {
            ensure_ray_entry_point(module);
            // Force a non-ray execution model but keep ray-only interfaces to trigger validation.
            if let Some(ep) = module.entry_points.first_mut() {
                if let Some(model) = ep.operands.get_mut(0) {
                    *model = ExecutionModel::Vertex.into();
                }
            }
        }
        InvalidKind::MissingRayCapability => {
            ensure_ray_entry_point(module);
            // Add a payload interface so the missing capability is observable.
            let int_ty = ensure_int_type(module);
            let payload_ptr =
                ensure_pointer_type(module, StorageClass::IncomingRayPayloadKHR, int_ty);
            let ids = insert_global_vars(
                module,
                payload_ptr,
                StorageClass::IncomingRayPayloadKHR,
                1,
            );
            push_interfaces(module, &ids);
            // Strip ray tracing capabilities if present.
            module.capabilities.retain(|inst| {
                match inst.operands.first() {
                    Some(Operand::Capability(spirv::Capability::RayTracingKHR))
                    | Some(Operand::Capability(spirv::Capability::RayTracingNV)) => false,
                    _ => true,
                }
            });
            // Keep Shader for a valid baseline.
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Capability::Shader.into()],
            ));
        }
        InvalidKind::RayPayloadTypeMismatch => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            // Deliberately use a payload variable that points to an int instead of a struct.
            let payload_ptr =
                ensure_pointer_type(module, StorageClass::IncomingRayPayloadKHR, int_ty);
            let ids = insert_global_vars(
                module,
                payload_ptr,
                StorageClass::IncomingRayPayloadKHR,
                1,
            );
            push_interfaces(module, &ids);
            // Ensure capability is present so the type error surfaces.
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Capability::RayTracingKHR.into()],
            ));
        }
        InvalidKind::CallableDataTypeMismatch => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::IncomingCallableDataKHR, int_ty);
            let ids =
                insert_global_vars(module, ptr, StorageClass::IncomingCallableDataKHR, 1);
            push_interfaces(module, &ids);
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Capability::RayTracingKHR.into()],
            ));
        }
        InvalidKind::HitAttributeTypeMismatch => {
            ensure_ray_entry_point(module);
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::HitAttributeKHR, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::HitAttributeKHR, 1);
            push_interfaces(module, &ids);
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Capability::RayTracingKHR.into()],
            ));
        }
        InvalidKind::HitAttributeOnRayGen => {
            ensure_ray_entry_point(module);
            if let Some(ep) = module.entry_points.first_mut() {
                if let Some(model) = ep.operands.get_mut(0) {
                    *model = ExecutionModel::RayGenerationKHR.into();
                }
            }
            let int_ty = ensure_int_type(module);
            let ptr = ensure_pointer_type(module, StorageClass::HitAttributeKHR, int_ty);
            let ids = insert_global_vars(module, ptr, StorageClass::HitAttributeKHR, 1);
            push_interfaces(module, &ids);
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Capability::RayTracingKHR.into()],
            ));
        }
    }
}

fn ensure_header(module: &mut dr::Module) -> &mut dr::ModuleHeader {
    if module.header.is_none() {
        module.header = Some(dr::ModuleHeader {
            magic_number: spirv::MAGIC_NUMBER,
            version: ((spirv::MAJOR_VERSION as u32) << 16) | ((spirv::MINOR_VERSION as u32) << 8),
            generator: 0,
            bound: 1,
            reserved_word: 0,
        });
    }
    module.header.as_mut().unwrap()
}

fn fresh_id(module: &mut dr::Module) -> u32 {
    let header = ensure_header(module);
    let id = header.bound;
    header.bound += 1;
    id
}

fn ensure_nop_present(module: &mut Module) {
    let mut has_nop = false;
    for func in &module.functions {
        for block in &func.blocks {
            if block
                .instructions
                .iter()
                .any(|inst| inst.class.opcode == Op::Nop)
            {
                has_nop = true;
                break;
            }
        }
    }
    if !has_nop {
        if let Some(func) = module.functions.first_mut() {
            if let Some(block) = func.blocks.first_mut() {
                block
                    .instructions
                    .insert(0, Instruction::new(Op::Nop, None, None, Vec::new()));
            }
        }
    }
}

fn minimal_valid_module_with_tag(tag: u32) -> Module {
    let mut builder = Builder::new();
    builder.capability(Capability::Shader);
    builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = builder.type_void();
    let func_ty = builder.type_function(void, vec![]);
    let int_ty = builder.type_int(32, 0);
    let _tag_const = builder.constant_bit32(int_ty, tag);
    let func = builder
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("begin function");
    builder.begin_block(None).expect("label");
    let _ = builder.insert_into_block(
        InsertPoint::End,
        Instruction::new(Op::Nop, None, None, Vec::new()),
    );
    builder.ret().expect("ret");
    builder.end_function().expect("end function");
    builder.entry_point(ExecutionModel::Vertex, func, "main", Vec::new());
    builder.module()
}

pub(crate) fn minimal_valid_words() -> Vec<u32> {
    minimal_valid_module_with_tag(0).assemble()
}

fn mutate_variable_storage_class(module: &mut dr::Module) -> bool {
    for func in &mut module.functions {
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if inst.class.opcode == Op::Variable && !inst.operands.is_empty() {
                    inst.operands[0] = StorageClass::Private.into();
                    return true;
                }
            }
        }
    }
    false
}

fn insert_mismatched_variable(module: &mut dr::Module) {
    if module.functions.is_empty() {
        *module = minimal_valid_module_with_tag(0);
    }
    let int_ty = ensure_int_type(module);
    let ptr_fn = ensure_pointer_type(module, StorageClass::Function, int_ty);

    let var_id = {
        let header = ensure_header(module);
        let id = header.bound;
        header.bound += 1;
        id
    };
    if let Some(func) = module.functions.first_mut() {
        if let Some(block) = func.blocks.first_mut() {
            block.instructions.insert(
                0,
                Instruction::new(
                    Op::Variable,
                    Some(ptr_fn),
                    Some(var_id),
                    vec![StorageClass::UniformConstant.into()],
                ),
            );
        }
    }
}

fn widen_access_chain(module: &mut dr::Module) -> bool {
    for func in &mut module.functions {
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if inst.class.opcode == Op::AccessChain {
                    inst.operands.push(5u32.into());
                    return true;
                }
            }
        }
    }
    false
}

fn inject_bad_access_chain(module: &mut dr::Module) {
    let int_ty = ensure_int_type(module);
    let ptr_ty = ensure_pointer_type(module, StorageClass::Function, int_ty);
    let access_id = fresh_id(module);
    if let Some(func) = module.functions.first_mut() {
        if let Some(block) = func.blocks.first_mut() {
            block.instructions.insert(
                0,
                Instruction::new(
                    Op::AccessChain,
                    Some(ptr_ty),
                    Some(access_id),
                    vec![999u32.into(), 1u32.into(), 2u32.into()],
                ),
            );
        }
    }
}

fn ensure_uniform_vars(module: &mut dr::Module, count: usize) -> (u32, Vec<u32>) {
    let int_ty = ensure_int_type(module);
    let ptr_ty = ensure_pointer_type(module, StorageClass::UniformConstant, int_ty);

    let mut vars = Vec::new();
    for _ in 0..count {
        vars.push(fresh_id(module));
    }
    (ptr_ty, vars)
}

fn ensure_int_type(module: &mut dr::Module) -> u32 {
    module
        .types_global_values
        .iter()
        .find_map(|inst| {
            if inst.class.opcode == Op::TypeInt {
                inst.result_id
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            let id = fresh_id(module);
            module.types_global_values.push(Instruction::new(
                Op::TypeInt,
                None,
                Some(id),
                vec![32u32.into(), 0u32.into()],
            ));
            id
        })
}

fn ensure_pointer_type(module: &mut dr::Module, storage: StorageClass, target: u32) -> u32 {
    module
        .types_global_values
        .iter()
        .find_map(|inst| {
            if inst.class.opcode != Op::TypePointer {
                return None;
            }
            let [Operand::StorageClass(sc), Operand::IdRef(pointee)] = inst.operands.as_slice()
            else {
                return None;
            };
            if *sc == storage && *pointee == target {
                return inst.result_id;
            }
            None
        })
        .unwrap_or_else(|| {
            let id = fresh_id(module);
            module.types_global_values.push(Instruction::new(
                Op::TypePointer,
                None,
                Some(id),
                vec![storage.into(), target.into()],
            ));
            id
        })
}

fn insert_global_vars(
    module: &mut dr::Module,
    ptr_ty: u32,
    storage: StorageClass,
    count: usize,
) -> Vec<u32> {
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let id = fresh_id(module);
        module.types_global_values.push(Instruction::new(
            Op::Variable,
            Some(ptr_ty),
            Some(id),
            vec![storage.into()],
        ));
        ids.push(id);
    }
    ids
}

fn insert_interface_var(module: &mut dr::Module, storage: StorageClass) -> u32 {
    let int_ty = ensure_int_type(module);
    let ptr = ensure_pointer_type(module, storage, int_ty);
    let ids = insert_global_vars(module, ptr, storage, 1);
    ids[0]
}

fn ensure_ray_entry_point(module: &mut dr::Module) {
    if module.entry_points.is_empty() || module.functions.is_empty() {
        *module = minimal_valid_module_with_tag(0);
    }
    if let Some(ep) = module.entry_points.first_mut() {
        if let Some(model) = ep.operands.get_mut(0) {
            *model = ExecutionModel::RayGenerationKHR.into();
        }
    }
}

fn push_interfaces(module: &mut dr::Module, ids: &[u32]) {
    if let Some(ep) = module.entry_points.first_mut() {
        for id in ids {
            ep.operands.push((*id).into());
        }
    }
}
