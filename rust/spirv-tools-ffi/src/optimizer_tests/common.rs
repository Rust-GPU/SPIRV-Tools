//! Common test utilities for optimizer tests.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Loader};
use rspirv::spirv::{AddressingModel, FunctionControl, MemoryModel, Op};
use std::sync::{Mutex, MutexGuard};

use crate::optimizer::optimize_basic_block;

/// Global mutex to ensure environment variable tests don't interfere with each other.
pub static ENV_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that locks the environment mutex and clears optimizer overrides.
pub struct OptimizerEnvGuard {
    _lock: MutexGuard<'static, ()>,
}

impl OptimizerEnvGuard {
    pub fn new() -> Self {
        // Recover from poisoned mutex (which happens when a test panics)
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::clear_rust_optimizer_override();
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
        Self { _lock: lock }
    }
}

impl Drop for OptimizerEnvGuard {
    fn drop(&mut self) {
        crate::clear_rust_optimizer_override();
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
    }
}

impl Default for OptimizerEnvGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to build a simple SPIR-V module with a single function.
pub struct TestModuleBuilder {
    pub builder: Builder,
    pub void_ty: u32,
    pub int_ty: u32,
    pub uint_ty: u32,
    pub bool_ty: u32,
    pub float_ty: u32,
}

impl TestModuleBuilder {
    /// Create a new test module builder with common types.
    pub fn new() -> Self {
        let mut builder = Builder::new();
        builder.capability(rspirv::spirv::Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::Simple);

        let void_ty = builder.type_void();
        let int_ty = builder.type_int(32, 1); // signed
        let uint_ty = builder.type_int(32, 0); // unsigned
        let bool_ty = builder.type_bool();
        let float_ty = builder.type_float(32, None);

        Self {
            builder,
            void_ty,
            int_ty,
            uint_ty,
            bool_ty,
            float_ty,
        }
    }

    /// Begin a function with no parameters returning void.
    pub fn begin_void_function(&mut self) {
        let func_ty = self.builder.type_function(self.void_ty, vec![]);
        self.builder
            .begin_function(self.void_ty, None, FunctionControl::NONE, func_ty)
            .expect("begin function");
        self.builder.begin_block(None).expect("begin block");
    }

    /// Begin a function with specified parameter types.
    pub fn begin_function_with_params(&mut self, param_types: Vec<u32>) -> Vec<u32> {
        let func_ty = self
            .builder
            .type_function(self.void_ty, param_types.clone());
        self.builder
            .begin_function(self.void_ty, None, FunctionControl::NONE, func_ty)
            .expect("begin function");

        let params: Vec<u32> = param_types
            .iter()
            .map(|ty| self.builder.function_parameter(*ty).expect("param"))
            .collect();

        self.builder.begin_block(None).expect("begin block");
        params
    }

    /// End the function and assemble the module.
    pub fn finish(mut self) -> Vec<u32> {
        self.builder.ret().expect("ret");
        self.builder.end_function().expect("end function");
        self.builder.module().assemble()
    }

    /// Create a signed integer constant.
    pub fn const_i32(&mut self, value: i32) -> u32 {
        self.builder.constant_bit32(self.int_ty, value as u32)
    }

    /// Create an unsigned integer constant.
    pub fn const_u32(&mut self, value: u32) -> u32 {
        self.builder.constant_bit32(self.uint_ty, value)
    }

    /// Create a boolean true constant.
    pub fn const_true(&mut self) -> u32 {
        self.builder.constant_true(self.bool_ty)
    }

    /// Create a boolean false constant.
    pub fn const_false(&mut self) -> u32 {
        self.builder.constant_false(self.bool_ty)
    }

    /// Create a boolean constant.
    pub fn const_bool(&mut self, value: bool) -> u32 {
        if value {
            self.const_true()
        } else {
            self.const_false()
        }
    }

    /// Create a 32-bit float constant.
    pub fn const_f32(&mut self, value: f32) -> u32 {
        self.builder.constant_bit32(self.float_ty, value.to_bits())
    }
}

impl Default for TestModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of analyzing an optimized module.
pub struct OptimizedModule {
    pub module: rspirv::dr::Module,
}

impl OptimizedModule {
    /// Optimize the given SPIR-V words and parse the result.
    pub fn from_words(words: &[u32]) -> Result<Self, crate::optimizer::OptimizeError> {
        let optimized = optimize_basic_block(words)?;
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        Ok(Self {
            module: loader.module(),
        })
    }

    /// Check if the module contains any instruction with the given opcode.
    pub fn has_opcode(&self, opcode: Op) -> bool {
        self.module
            .all_inst_iter()
            .any(|inst| inst.class.opcode == opcode)
    }

    /// Count instructions with the given opcode.
    pub fn count_opcode(&self, opcode: Op) -> usize {
        self.module
            .all_inst_iter()
            .filter(|inst| inst.class.opcode == opcode)
            .count()
    }

    /// Check if the module contains a constant with the given value.
    pub fn has_constant_u32(&self, value: u32) -> bool {
        self.module.all_inst_iter().any(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(value)]
        })
    }
}
