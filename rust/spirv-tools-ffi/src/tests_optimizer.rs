#[cfg(test)]
mod optimizer_tests {
    use crate::optimize_basic_block as optimize_wrapped_block;
    use crate::optimizer::optimize_basic_block;
    use rspirv::binary::Assemble;
    use rspirv::dr::{Builder, Loader};
    use rspirv::spirv::{FunctionControl, Op};
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct OptimizerEnvGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl OptimizerEnvGuard {
        fn new() -> Self {
            let lock = ENV_LOCK.lock().expect("env mutex poisoned");
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

    #[test]
    fn optimizer_reports_parse_error() {
        let _guard = OptimizerEnvGuard::new();
        let invalid_words = vec![0u32]; // not a valid module header
        let result = optimize_wrapped_block(&invalid_words);
        assert!(!result.success);
        assert!(matches!(
            result.error,
            crate::OptimizeError::Parse | crate::OptimizeError::Optimize
        ));
    }

    #[test]
    fn optimizer_reports_disabled_kind() {
        let _guard = OptimizerEnvGuard::new();
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");

        // Build a simple module that would otherwise be optimized.
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let _ = b.i_add(int, None, c2, c3);
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let result = optimize_wrapped_block(&words);
        assert!(result.success, "disable should passthrough successfully");
        assert_eq!(result.error, crate::OptimizeError::Disabled);
        assert_eq!(result.words, words, "disable should leave module unchanged");
    }

    #[test]
    fn optimizer_basic_block_pass_through_non_arith() {
        let mut b = Builder::new();
        let void = b.type_void();
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        b.begin_block(None).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let result = optimize_basic_block(&words).expect("optimization should succeed");
        assert_eq!(result, words);
    }

    #[test]
    fn optimizer_folds_constant_add_block() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let sum = b.i_add(int, None, c2, c3).expect("add id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_const_five = false;
        let mut has_add = false;
        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5u32)]
                        && inst.result_id == Some(sum)
                    {
                        found_const_five = true;
                    }
                }
                Op::IAdd => has_add = true,
                _ => {}
            }
        }
        assert!(found_const_five, "optimizer should fold to a const 5");
        assert!(!has_add, "addition should be folded away");
    }

    #[test]
    fn optimizer_folds_sub_self_to_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c7 = b.constant_bit32(int, 7);
        let sub = b.i_sub(int, None, c7, c7).expect("sub id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_zero = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(sub)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_zero = true;
            }
            assert_ne!(inst.class.opcode, Op::ISub, "sub should fold away");
        }
        assert!(found_zero, "sub should fold to constant zero");
    }

    fn build_factored_mul_sum_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let param = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let mul_left = b.i_mul(int, None, param, c2).expect("mul left");
        let mul_right = b.i_mul(int, None, param, c3).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add, param)
    }

    fn build_factored_const_mul_sum_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, c4).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add)
    }

    fn build_factored_const_mul_sum_commuted_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, c4, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add)
    }

    fn build_factored_const_mul_sub_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, c4).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_const_mul_sub_commuted_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, c4, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_const_mul_sum_mixed_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
        let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add)
    }

    fn build_factored_const_mul_sub_mixed_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let mul_left = b.i_mul(int, None, lhs, c4).expect("mul left");
        let mul_right = b.i_mul(int, None, c4, rhs).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_mixed_const_difference_mul_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c5 = b.constant_bit32(int, 5);
        let mul_left = b.i_mul(int, None, c2, x).expect("mul left");
        let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_difference_mul_commuted_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c5 = b.constant_bit32(int, 5);
        let mul_left = b.i_mul(int, None, x, c2).expect("mul left commuted");
        let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_positive_difference_mul_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c7 = b.constant_bit32(int, 7);
        let c2 = b.constant_bit32(int, 2);
        let mul_left = b.i_mul(int, None, c7, x).expect("mul left");
        let mul_right = b.i_mul(int, None, c2, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_positive_difference_mul_commuted_module() -> (Vec<u32>, u32, u32)
    {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c7 = b.constant_bit32(int, 7);
        let c2 = b.constant_bit32(int, 2);
        let mul_left = b.i_mul(int, None, x, c7).expect("mul left commuted");
        let mul_right = b.i_mul(int, None, c2, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_wrap_negative_difference_mul_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let high = b.constant_bit32(int, u32::MAX - 1);
        let c5 = b.constant_bit32(int, 5);
        let mul_left = b.i_mul(int, None, high, x).expect("mul left");
        let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_wrap_negative_difference_mul_commuted_module(
    ) -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let high = b.constant_bit32(int, u32::MAX - 1);
        let c5 = b.constant_bit32(int, 5);
        let mul_left = b.i_mul(int, None, x, high).expect("mul left commuted");
        let mul_right = b.i_mul(int, None, c5, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_wrap_positive_difference_mul_module() -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let neg1 = b.constant_bit32(int, u32::MAX);
        let neg4 = b.constant_bit32(int, u32::MAX - 3);
        let mul_left = b.i_mul(int, None, neg1, x).expect("mul left");
        let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_mixed_const_wrap_positive_difference_mul_commuted_module(
    ) -> (Vec<u32>, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let neg1 = b.constant_bit32(int, u32::MAX);
        let neg4 = b.constant_bit32(int, u32::MAX - 3);
        let mul_left = b.i_mul(int, None, x, neg1).expect("mul left commuted");
        let mul_right = b.i_mul(int, None, neg4, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, x)
    }

    fn build_factored_const_equal_difference_mul_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c6 = b.constant_bit32(int, 6);
        let mul_left = b.i_mul(int, None, c6, x).expect("mul left");
        let mul_right = b.i_mul(int, None, c6, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_const_equal_difference_mul_commuted_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c6 = b.constant_bit32(int, 6);
        let mul_left = b.i_mul(int, None, x, c6).expect("mul left commuted");
        let mul_right = b.i_mul(int, None, c6, x).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_const_equal_difference_unsigned_mul_module() -> (Vec<u32>, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let x = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let c6 = b.constant_bit32(int, 6);
        let mul_left = b.i_mul(int, None, c6, x).expect("mul left");
        let mul_right = b.i_mul(int, None, x, c6).expect("mul right commuted");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub)
    }

    fn build_factored_symbolic_mul_sub_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, base, rhs).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, base, lhs, rhs)
    }

    fn build_factored_symbolic_mul_sub_mixed_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, base, lhs, rhs)
    }

    fn build_factored_symbolic_mul_add_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, base, rhs).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add, base, lhs, rhs)
    }

    fn build_factored_symbolic_mul_add_mixed_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, base, lhs).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add, base, lhs, rhs)
    }

    fn build_factored_symbolic_mul_sub_commuted_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, lhs, base).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
        let sub = b.i_sub(int, None, mul_left, mul_right).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), sub, base, lhs, rhs)
    }

    fn build_factored_symbolic_mul_add_commuted_module() -> (Vec<u32>, u32, u32, u32, u32) {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![int, int, int]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        b.begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let base = b.function_parameter(int).unwrap();
        let lhs = b.function_parameter(int).unwrap();
        let rhs = b.function_parameter(int).unwrap();
        b.begin_block(None).unwrap();
        let mul_left = b.i_mul(int, None, lhs, base).expect("mul left");
        let mul_right = b.i_mul(int, None, rhs, base).expect("mul right");
        let add = b.i_add(int, None, mul_left, mul_right).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        (b.module().assemble(), add, base, lhs, rhs)
    }

    #[test]
    fn optimizer_factors_common_multiplicand() {
        let (words, add_id, param_id) = build_factored_mul_sum_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut mul_count = 0;
        let mut add_present = false;
        let mut factored = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param_id || rhs == param_id;
                    let const_id = if lhs == param_id { rhs } else { lhs };
                    let is_const_five = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 5)
                        .unwrap_or(false);
                    factored = uses_param && is_const_five;
                }
                Op::IAdd => add_present = true,
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "factored multiply should reuse add id and use 5 * param"
        );
        assert!(!add_present, "addition should be removed after factoring");
    }

    #[test]
    fn optimizer_factors_shared_constant_from_sum() {
        let (words, add_id) = build_factored_const_mul_sum_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut add_result = None;
        let mut scaling_count = 0;
        let mut factored = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IAdd => {
                    add_result = inst.result_id;
                }
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id || rhs == add_res_id;
                    let const_id = if lhs == add_res_id { rhs } else { lhs };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_add && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id;
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_add && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(
            scaling_count, 1,
            "factoring should leave one scaling instruction"
        );
        assert!(add_result.is_some(), "addition should remain as inner sum");
        assert!(
            factored,
            "factored multiply should reuse add result and multiply by four"
        );
    }

    #[test]
    fn optimizer_factors_shared_constant_from_sum_commuted_mul() {
        let (words, add_id) = build_factored_const_mul_sum_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut add_result = None;
        let mut scaling_count = 0;
        let mut factored = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IAdd => {
                    add_result = inst.result_id;
                }
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id || rhs == add_res_id;
                    let const_id = if lhs == add_res_id { rhs } else { lhs };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_add && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id;
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_add && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(
            scaling_count, 1,
            "factoring should leave one scaling instruction"
        );
        assert!(add_result.is_some(), "addition should remain as inner sum");
        assert!(
            factored,
            "factored multiply should reuse add result when constants lead the multiplies"
        );
    }

    #[test]
    fn optimizer_factors_shared_constant_from_sum_mixed_mul_order() {
        let (words, add_id) = build_factored_const_mul_sum_mixed_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut add_result = None;
        let mut scaling_count = 0;
        let mut factored = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IAdd => {
                    add_result = inst.result_id;
                }
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id || rhs == add_res_id;
                    let const_id = if lhs == add_res_id { rhs } else { lhs };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_add && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(add_res_id) = add_result else {
                        continue;
                    };
                    let uses_add = lhs == add_res_id;
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_add && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(
            scaling_count, 1,
            "factoring should leave one scaling instruction"
        );
        assert!(add_result.is_some(), "addition should remain as inner sum");
        assert!(
            factored,
            "factored multiply should reuse add result when only one multiply commutes the constant"
        );
    }

    #[test]
    fn optimizer_factors_mixed_constant_difference_into_single_mul() {
        let (words, sub_id, param) = build_factored_mixed_const_difference_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, u32::MAX - 2, "expected -3 as two's complement");
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_constant_difference_commuted_into_single_mul() {
        let (words, sub_id, param) = build_factored_mixed_const_difference_mul_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, u32::MAX - 2, "expected -3 as two's complement");
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_positive_constant_difference_into_single_mul() {
        let (words, sub_id, param) = build_factored_mixed_const_positive_difference_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, 5);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_positive_constant_difference_commuted_into_single_mul() {
        let (words, sub_id, param) =
            build_factored_mixed_const_positive_difference_mul_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, 5);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_wrap_negative_constant_difference_into_single_mul() {
        let (words, sub_id, param) =
            build_factored_mixed_const_wrap_negative_difference_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, u32::MAX - 6);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_wrap_negative_constant_difference_commuted_into_single_mul() {
        let (words, sub_id, param) =
            build_factored_mixed_const_wrap_negative_difference_mul_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, u32::MAX - 6);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_wrap_positive_constant_difference_into_single_mul() {
        let (words, sub_id, param) =
            build_factored_mixed_const_wrap_positive_difference_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, 3);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_mixed_wrap_positive_constant_difference_commuted_into_single_mul() {
        let (words, sub_id, param) =
            build_factored_mixed_const_wrap_positive_difference_mul_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut factored = false;
        let mut constant = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IMul => {
                    mul_count += 1;
                    assert_eq!(
                        inst.result_id,
                        Some(sub_id),
                        "mul should replace subtract result"
                    );
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_param = lhs == param || rhs == param;
                    let const_id = if lhs == param { rhs } else { lhs };
                    constant = Some(const_id);
                    factored = uses_param;
                }
                Op::Constant => {
                    if let Some(value) = inst.operands.first().and_then(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    }) {
                        if Some(inst.result_id.unwrap()) == constant {
                            assert_eq!(value, 3);
                        }
                    }
                }
                Op::ISub => panic!("subtract should fold into a single multiply"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(
            factored,
            "mul should reuse subtract id with the shared param"
        );
        assert!(
            constant.is_some(),
            "factored multiply should include the constant difference"
        );
    }

    #[test]
    fn optimizer_factors_equal_constant_difference_into_zero() {
        let (words, sub_id) = build_factored_const_equal_difference_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut saw_zero = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
                Op::Constant => {
                    if inst.result_id == Some(sub_id) {
                        if let Some(value) = inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }) {
                            assert_eq!(value, 0);
                            saw_zero = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(
            saw_zero,
            "subtract id should be replaced by a zero constant after factoring"
        );
    }

    #[test]
    fn optimizer_factors_equal_constant_difference_commuted_into_zero() {
        let (words, sub_id) = build_factored_const_equal_difference_mul_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut saw_zero = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
                Op::Constant => {
                    if inst.result_id == Some(sub_id) {
                        if let Some(value) = inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }) {
                            assert_eq!(value, 0);
                            saw_zero = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(
            saw_zero,
            "subtract id should be replaced by a zero constant after factoring"
        );
    }

    #[test]
    fn optimizer_factors_equal_constant_difference_unsigned_into_zero() {
        let (words, sub_id) = build_factored_const_equal_difference_unsigned_mul_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut saw_zero = false;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub | Op::IMul => panic!("subtract/multiply should fold away"),
                Op::Constant => {
                    if inst.result_id == Some(sub_id) {
                        if let Some(value) = inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }) {
                            assert_eq!(value, 0);
                            saw_zero = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(
            saw_zero,
            "subtract id should be replaced by a zero constant after factoring"
        );
    }
    #[test]
    fn optimizer_factors_shared_constant_from_sub() {
        let (words, sub_id) = build_factored_const_mul_sub_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut sub_count = 0;
        let mut scaling_count = 0;
        let mut factored = false;
        let mut inner_sub = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IAdd => panic!("addition should be factored into the subtract branch"),
                Op::ISub => {
                    sub_count += 1;
                    inner_sub = inst.result_id;
                }
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                    let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                        rhs
                    } else {
                        lhs
                    };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_sub && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_sub && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(sub_count, 1, "inner subtract should remain");
        assert_eq!(
            scaling_count, 1,
            "factoring should leave a single scaling op"
        );
        assert!(factored, "scaling should reuse the subtract result id");
    }

    #[test]
    fn optimizer_factors_shared_constant_from_sub_commuted_mul() {
        let (words, sub_id) = build_factored_const_mul_sub_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut sub_count = 0;
        let mut scaling_count = 0;
        let mut factored = false;
        let mut inner_sub = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::ISub => {
                    sub_count += 1;
                    inner_sub = inst.result_id;
                }
                Op::IAdd => panic!("addition should be factored into the subtract branch"),
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                    let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                        rhs
                    } else {
                        lhs
                    };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_sub && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_sub && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(sub_count, 1, "inner subtract should remain");
        assert_eq!(
            scaling_count, 1,
            "factoring should leave a single scaling op"
        );
        assert!(
            factored,
            "scaling should reuse the subtract result id when constants lead the multiplies"
        );
    }

    #[test]
    fn optimizer_factors_shared_constant_from_sub_mixed_mul_order() {
        let (words, sub_id) = build_factored_const_mul_sub_mixed_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut constants = std::collections::HashMap::new();
        let mut sub_count = 0;
        let mut scaling_count = 0;
        let mut factored = false;
        let mut inner_sub = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if let (Some(id), Some(value)) = (
                        inst.result_id,
                        inst.operands.first().and_then(|op| match op {
                            rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                            _ => None,
                        }),
                    ) {
                        constants.insert(id, value);
                    }
                }
                Op::IAdd => panic!("addition should not remain after factoring the subtract"),
                Op::ISub => {
                    sub_count += 1;
                    inner_sub = inst.result_id;
                }
                Op::IMul => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid || rhs == sid);
                    let const_id = if inner_sub.is_some_and(|sid| lhs == sid) {
                        rhs
                    } else {
                        lhs
                    };
                    let is_const_four = constants
                        .get(&const_id)
                        .copied()
                        .map(|v| v == 4)
                        .unwrap_or(false);
                    factored = uses_sub && is_const_four;
                }
                Op::ShiftLeftLogical => {
                    scaling_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_sub = inner_sub.is_some_and(|sid| lhs == sid);
                    let is_shift_two = constants
                        .get(&rhs)
                        .copied()
                        .map(|v| v == 2)
                        .unwrap_or(false);
                    factored = uses_sub && is_shift_two;
                }
                _ => {}
            }
        }

        assert_eq!(sub_count, 1, "inner subtract should remain");
        assert_eq!(
            scaling_count, 1,
            "factoring should leave a single scaling op"
        );
        assert!(
            factored,
            "scaling should reuse the subtract result id even when only one mul commutes the constant"
        );
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_sub() {
        let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut sub_seen = false;
        let mut factored = false;
        let mut diff_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub => {
                    diff_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        sub_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && diff_id == Some(diff_operand);
                }
                Op::IAdd => panic!("addition should not remain for subtract factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(sub_seen, "inner subtraction should remain");
        assert!(factored, "mul should reuse sub id and use base*diff");
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_add() {
        let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut add_seen = false;
        let mut factored = false;
        let mut sum_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IAdd => {
                    sum_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        add_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && sum_id == Some(sum_operand);
                }
                Op::ISub => panic!("subtraction should not remain in addition factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(add_seen, "inner addition should remain");
        assert!(factored, "mul should reuse add id and use base*sum");
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_sub_commuted_mul() {
        let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut sub_seen = false;
        let mut factored = false;
        let mut diff_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub => {
                    diff_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        sub_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && diff_id == Some(diff_operand);
                }
                Op::IAdd => panic!("addition should not remain for subtract factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(sub_seen, "inner subtraction should remain");
        assert!(
            factored,
            "mul should reuse sub id and keep the base multiplicand regardless of order"
        );
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_sub_mixed_mul_order() {
        let (words, sub_id, base, lhs, rhs) = build_factored_symbolic_mul_sub_mixed_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut sub_seen = false;
        let mut factored = false;
        let mut diff_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::ISub => {
                    diff_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        sub_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(sub_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let diff_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && diff_id == Some(diff_operand);
                }
                Op::IAdd => panic!("addition should not remain for subtract factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(sub_seen, "inner subtraction should remain");
        assert!(
            factored,
            "mul should reuse sub id and keep the base multiplicand when only one mul commutes operands"
        );
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_add_commuted_mul() {
        let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_commuted_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut add_seen = false;
        let mut factored = false;
        let mut sum_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IAdd => {
                    sum_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        add_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && sum_id == Some(sum_operand);
                }
                Op::ISub => panic!("subtraction should not remain in addition factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(add_seen, "inner addition should remain");
        assert!(
            factored,
            "mul should reuse add id and keep the base multiplicand regardless of order"
        );
    }

    #[test]
    fn optimizer_factors_symbolic_multiplicand_from_add_mixed_mul_order() {
        let (words, add_id, base, lhs, rhs) = build_factored_symbolic_mul_add_mixed_module();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut mul_count = 0;
        let mut add_seen = false;
        let mut factored = false;
        let mut sum_id = None;

        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IAdd => {
                    sum_id = inst.result_id;
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    if (lhs_id == lhs && rhs_id == rhs) || (lhs_id == rhs && rhs_id == lhs) {
                        add_seen = true;
                    }
                }
                Op::IMul => {
                    mul_count += 1;
                    if inst.result_id != Some(add_id) {
                        continue;
                    }
                    let Some(lhs_id) = inst.operands.get(0).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let Some(rhs_id) = inst.operands.get(1).and_then(|op| op.id_ref_any()) else {
                        continue;
                    };
                    let uses_base = lhs_id == base || rhs_id == base;
                    let sum_operand = if lhs_id == base { rhs_id } else { lhs_id };
                    factored = uses_base && sum_id == Some(sum_operand);
                }
                Op::ISub => panic!("subtraction should not remain in addition factoring"),
                _ => {}
            }
        }

        assert_eq!(mul_count, 1, "factoring should leave one multiply");
        assert!(add_seen, "inner addition should remain");
        assert!(
            factored,
            "mul should reuse add id and keep the base multiplicand when only one mul commutes operands"
        );
    }

    #[test]
    fn optimizer_folds_mul_by_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c5 = b.constant_bit32(int, 5);
        let c0 = b.constant_bit32(int, 0);
        let mul = b.i_mul(int, None, c5, c0).expect("mul id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_zero = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::IMul {
                panic!("mul should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(mul)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_zero = true;
            }
        }
        assert!(found_zero, "mul by zero should fold to constant zero");
    }

    #[test]
    fn optimizer_folds_mul_by_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c7 = b.constant_bit32(int, 7);
        let c1 = b.constant_bit32(int, 1);
        let mul = b.i_mul(int, None, c7, c1).expect("mul id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_const = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::IMul {
                panic!("mul should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(mul)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "mul by one should fold to original value");
    }

    #[test]
    fn optimizer_rewrites_band_pow2_mask_to_umod() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        // x = 5 + 6, then x & 7.
        let c5 = b.constant_bit32(int, 2);
        let c6 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c5, c6).expect("add id");
        let mask = b.constant_bit32(int, 7);
        let _band = b.bitwise_and(int, None, add, mask).expect("bitwise and id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut has_umod_pow2 = false;
        let mut has_const_three = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::BitwiseAnd {
                panic!("bitwise and should be eliminated");
            }
            if inst.class.opcode == Op::UMod
                && inst
                    .operands
                    .iter()
                    .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(value) if *value == 8))
            {
                has_umod_pow2 = true;
            }
            if inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(3)]
            {
                has_const_three = true;
            }
        }
        assert!(
            has_umod_pow2 || has_const_three,
            "expected band mask to become x % 8 or a folded constant"
        );
    }

    #[test]
    fn optimizer_simplifies_bitand_all_ones() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let value = b.constant_bit32(int, 0x1234_5678);
        let ones = b.constant_bit32(int, u32::MAX);
        let band = b.bitwise_and(int, None, value, ones).expect("band id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseAnd,
                "and with all ones should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(band)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0x1234_5678)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "and with all ones should keep the original value"
        );
    }

    #[test]
    fn optimizer_simplifies_bitor_all_ones() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let value = b.constant_bit32(int, 0xBEEF_CAFE);
        let ones = b.constant_bit32(int, u32::MAX);
        let bor = b.bitwise_or(int, None, value, ones).expect("bor id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseOr,
                "or with all ones should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "or with all ones should fold to an all-ones constant"
        );
    }

    #[test]
    fn optimizer_rewrites_bitxor_all_ones_to_not() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 5);
        let rhs = b.constant_bit32(int, 6);
        let input = b.i_add(int, None, lhs, rhs).expect("value");
        let ones = b.constant_bit32(int, u32::MAX);
        let bxor = b.bitwise_xor(int, None, input, ones).expect("xor id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let expected = !11u32;
        let mut saw_not = false;
        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            match inst.class.opcode {
                Op::BitwiseXor => panic!("xor with all ones should be rewritten"),
                Op::Not if inst.result_id == Some(bxor) => saw_not = true,
                Op::Constant
                    if inst.result_id == Some(bxor)
                        && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(expected)] =>
                {
                    saw_const = true;
                }
                _ => {}
            }
        }
        assert!(
            saw_not || saw_const,
            "xor with all ones should lower to a bitwise not (or folded constant) with the same id"
        );
    }

    #[test]
    fn optimizer_folds_bitand_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let value = b.constant_bit32(int, 0xDEAD_BEEF);
        let zero = b.constant_bit32(int, 0);
        let band = b.bitwise_and(int, None, value, zero).expect("and");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_zero = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseAnd,
                "and with zero should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(band)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                saw_zero = true;
            }
        }
        assert!(saw_zero, "and with zero should fold to zero with same id");
    }

    #[test]
    fn optimizer_folds_bitor_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 2);
        let rhs = b.constant_bit32(int, 3);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let zero = b.constant_bit32(int, 0);
        let bor = b.bitwise_or(int, None, value, zero).expect("or");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseOr,
                "or with zero should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "or with zero should fold to original value with same id"
        );
    }

    #[test]
    fn optimizer_folds_bitxor_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 7);
        let rhs = b.constant_bit32(int, 9);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let zero = b.constant_bit32(int, 0);
        let bxor = b.bitwise_xor(int, None, value, zero).expect("xor");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseXor,
                "xor with zero should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bxor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(16)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "xor with zero should fold to the original value with same id"
        );
    }

    #[test]
    fn optimizer_folds_bitand_self() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 4);
        let rhs = b.constant_bit32(int, 6);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let band = b.bitwise_and(int, None, value, value).expect("and");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseAnd,
                "and with self should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(band)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(10)]
            {
                saw_const = true;
            }
        }
        assert!(saw_const, "and with self should fold to the original value");
    }

    #[test]
    fn optimizer_folds_bitor_self() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 7);
        let rhs = b.constant_bit32(int, 8);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let bor = b.bitwise_or(int, None, value, value).expect("or");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseOr,
                "or with self should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(15)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "or with self should fold to the original value with same id"
        );
    }

    #[test]
    fn optimizer_folds_bitxor_self() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 5);
        let rhs = b.constant_bit32(int, 9);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let bxor = b.bitwise_xor(int, None, value, value).expect("xor");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseXor,
                "xor with self should be eliminated"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bxor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                saw_const = true;
            }
        }
        assert!(saw_const, "xor with self should fold to zero with same id");
    }

    #[test]
    fn optimizer_folds_bitand_complement() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 2);
        let rhs = b.constant_bit32(int, 3);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let not_value = b.not(int, None, value).expect("not");
        let band = b.bitwise_and(int, None, value, not_value).expect("and");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseAnd,
                "and with complement should be eliminated"
            );
            assert_ne!(inst.class.opcode, Op::Not, "dead not should be removed");
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(band)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "and with complement should fold to zero with same id"
        );
    }

    #[test]
    fn optimizer_folds_bitor_complement() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let lhs = b.constant_bit32(int, 11);
        let rhs = b.constant_bit32(int, 5);
        let value = b.i_add(int, None, lhs, rhs).expect("value");
        let not_value = b.not(int, None, value).expect("not");
        let bor = b.bitwise_or(int, None, value, not_value).expect("or");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseOr,
                "or with complement should be eliminated"
            );
            assert_ne!(inst.class.opcode, Op::Not, "dead not should be removed");
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(bor)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(u32::MAX)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_const,
            "or with complement should fold to all ones with same id"
        );
    }

    #[test]
    fn optimizer_factors_linear_combination_into_single_mul() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let factor = b.constant_bit32(int, 4);
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let mul1 = b.i_mul(int, None, factor, c2).expect("mul1");
        let mul2 = b.i_mul(int, None, c3, factor).expect("mul2");
        let add = b.i_add(int, None, mul1, mul2).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_const = false;
        for inst in optimized_module.all_inst_iter() {
            match inst.class.opcode {
                Op::IAdd => panic!("addition should be factored away"),
                Op::Constant => {
                    if inst.result_id == Some(add)
                        && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(20u32)]
                    {
                        found_const = true;
                    }
                }
                _ => {}
            }
        }
        assert!(found_const, "should fold to single constant result");
    }

    #[test]
    fn optimizer_strength_reduces_mul_pow2() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c8 = b.constant_bit32(int, 8);
        let c2 = b.constant_bit32(int, 2);
        // Keep operands non-constant to avoid full folding to a literal.
        let mul = b.i_mul(int, None, c2, c8).expect("mul");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();
        let mut found_shift = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::ShiftLeftLogical && inst.result_id == Some(mul) {
                found_shift = true;
            }
            assert_ne!(
                inst.class.opcode,
                Op::IMul,
                "mul by power of two should rewrite"
            );
        }
        // Allow either shift or folded constant in case both operands are const.
        let found_const = module.all_inst_iter().any(|inst| {
            inst.class.opcode == Op::Constant
                && inst.result_id == Some(mul)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(16)]
        });
        assert!(
            found_shift || found_const,
            "mul by pow2 should strength-reduce or fold"
        );
        // Disable optimizer and ensure passthrough works.
        let _env = OptimizerEnvGuard::new();
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
        let passthrough = optimize_wrapped_block(&words);
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
        assert_eq!(
            passthrough.words, words,
            "disable flag should skip optimization"
        );
    }

    #[test]
    fn optimizer_eliminates_shift_by_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c4 = b.constant_bit32(int, 4);
        let zero = b.constant_bit32(int, 0);
        let shl = b.shift_left_logical(int, None, c4, zero).expect("shift id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::ShiftLeftLogical,
                "shift by zero should be removed"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(shl)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(4)]
            {
                found_const = true;
            }
        }
        assert!(
            found_const,
            "shift by zero should reuse original id as constant"
        );
    }

    #[test]
    fn optimizer_folds_rotate_pattern() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let value = b.constant_bit32(int, 0x12);
        let shift = b.constant_bit32(int, 3);
        let left = b.shift_left_logical(int, None, value, shift).expect("shl");
        let right_amount = b.constant_bit32(int, 29);
        let right = b
            .shift_right_logical(int, None, value, right_amount)
            .expect("shr");
        let or = b
            .bitwise_or(int, None, left, right)
            .expect("rotate pattern");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            assert_ne!(
                inst.class.opcode,
                Op::BitwiseOr,
                "rotate pattern OR should be folded away"
            );
            assert_ne!(
                inst.class.opcode,
                Op::ShiftLeftLogical,
                "rotate left shift should be folded away"
            );
            assert_ne!(
                inst.class.opcode,
                Op::ShiftRightLogical,
                "rotate right shift should be folded away"
            );
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(or)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0x90)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "rotate pattern should fold to constant 0x90");
    }

    #[test]
    fn optimizer_cancels_add_sub_chain() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let ca = b.constant_bit32(int, 42);
        let cb = b.constant_bit32(int, 5);
        let sub = b.i_sub(int, None, ca, cb).expect("sub");
        let add = b.i_add(int, None, sub, cb).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_const = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::IAdd || inst.class.opcode == Op::ISub {
                panic!("add/sub chain should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(add)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(42)]
            {
                found_const = true;
            }
        }
        assert!(
            found_const,
            "add-sub cancellation should fold to original value"
        );
    }

    #[test]
    fn optimizer_folds_add_with_negate() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c5 = b.constant_bit32(int, 5);
        let neg = b.s_negate(int, None, c5).expect("neg");
        let add = b.i_add(int, None, c5, neg).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut found_zero = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::IAdd {
                panic!("add should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(add)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_zero = true;
            }
        }
        assert!(found_zero, "add with negate should fold to zero");
    }

    #[test]
    fn optimizer_rewrites_neg_sub_to_swap() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let ca = b.constant_bit32(int, 10);
        let cb = b.constant_bit32(int, 3);
        let sub = b.i_sub(int, None, ca, cb).expect("sub");
        let neg = b.s_negate(int, None, sub).expect("neg");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut folded_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(neg)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(7))]
            {
                folded_const = true;
            }
        }
        assert!(
            folded_const,
            "negated subtraction should fold to constant -7 (b - a)"
        );
    }

    #[test]
    fn optimizer_folds_udiv_by_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c9 = b.constant_bit32(int, 9);
        let c1 = b.constant_bit32(int, 1);
        let div = b.u_div(int, None, c9, c1).expect("udiv");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::UDiv {
                panic!("udiv by one should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(div)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "udiv by one should fold to the original value");
    }

    #[test]
    fn optimizer_folds_sdiv_by_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c9 = b.constant_bit32(int, 9);
        let c1 = b.constant_bit32(int, 1);
        let div = b.s_div(int, None, c9, c1).expect("sdiv");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::SDiv {
                panic!("sdiv by one should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(div)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(9)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "sdiv by one should fold to the original value");
    }

    #[test]
    fn optimizer_folds_urem_by_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c9 = b.constant_bit32(int, 9);
        let c1 = b.constant_bit32(int, 1);
        let rem = b.u_mod(int, None, c9, c1).expect("urem");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::UMod {
                panic!("urem by one should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(rem)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "urem by one should fold to zero");
    }

    #[test]
    fn optimizer_folds_srem_by_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c9 = b.constant_bit32(int, 9);
        let c1 = b.constant_bit32(int, 1);
        let rem = b.s_rem(int, None, c9, c1).expect("srem");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut found_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == Op::SRem {
                panic!("srem by one should be folded away");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(rem)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                found_const = true;
            }
        }
        assert!(found_const, "srem by one should fold to zero");
    }

    #[test]
    fn optimizer_folds_mul_by_neg_one() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 1);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c6 = b.constant_bit32(int, 6);
        let c_neg_one = b.constant_bit32(int, u32::MAX);
        let mul = b.i_mul(int, None, c6, c_neg_one).expect("mul id");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut folded = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::IMul {
                panic!("mul by -1 should be rewritten or folded");
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(mul)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0u32.wrapping_sub(6))]
            {
                folded = true;
            }
            if inst.class.opcode == Op::SNegate
                && inst.result_id == Some(mul)
                && inst.operands == vec![rspirv::dr::Operand::IdRef(c6)]
            {
                folded = true;
            }
        }
        assert!(folded, "mul by -1 should become negate or folded const");
    }

    #[test]
    fn optimizer_respects_disable_env() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c2, c3).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let _env = OptimizerEnvGuard::new();
        crate::clear_rust_optimizer_override();
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
        let optimized = optimize_wrapped_block(&words);
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");

        assert!(optimized.success, "wrapper should not error");
        assert_eq!(optimized.words, words, "disable env should passthrough");

        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized.words, &mut loader).expect("parse optimized");
        let module = loader.module();
        let mut saw_add = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == rspirv::spirv::Op::IAdd && inst.result_id == Some(add) {
                saw_add = true;
            }
        }
        assert!(saw_add, "add should remain when optimizer is disabled");
    }

    #[test]
    fn optimizer_force_env_is_cleared_with_override_reset() {
        let _env = OptimizerEnvGuard::new();
        crate::set_rust_optimizer_override(true);
        assert_eq!(
            std::env::var("SPIRV_TOOLS_FORCE_RUST_OPT").as_deref(),
            Ok("1")
        );
        crate::clear_rust_optimizer_override();
        assert!(std::env::var("SPIRV_TOOLS_FORCE_RUST_OPT").is_err());
    }

    #[test]
    fn optimizer_disable_env_wins_over_force_env() {
        let _env = OptimizerEnvGuard::new();
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
        std::env::set_var("SPIRV_TOOLS_FORCE_RUST_OPT", "1");

        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c2, c3).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_wrapped_block(&words);

        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");
        std::env::remove_var("SPIRV_TOOLS_FORCE_RUST_OPT");

        assert!(optimized.success, "wrapper should not error");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized.words, &mut loader).expect("parse optimized");
        let module = loader.module();
        let mut saw_add = false;
        let mut found_const_five = false;
        for inst in module.all_inst_iter() {
            match inst.class.opcode {
                rspirv::spirv::Op::IAdd if inst.result_id == Some(add) => saw_add = true,
                rspirv::spirv::Op::Constant => {
                    if inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5u32)]
                        && inst.result_id == Some(add)
                    {
                        found_const_five = true;
                    }
                }
                _ => {}
            }
        }
        assert!(
            saw_add && !found_const_five,
            "disable env should win even when force env is set"
        );
    }

    #[test]
    fn optimizer_override_can_disable_even_without_env() {
        let _env = OptimizerEnvGuard::new();
        crate::set_rust_optimizer_override(false);
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c2, c3).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_wrapped_block(&words);
        crate::clear_rust_optimizer_override();

        assert!(optimized.success, "wrapper should not error");
        assert_eq!(
            optimized.words, words,
            "override disable should passthrough"
        );
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized.words, &mut loader).expect("parse optimized");
        let module = loader.module();
        let mut saw_add = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == rspirv::spirv::Op::IAdd && inst.result_id == Some(add) {
                saw_add = true;
            }
        }
        assert!(
            saw_add,
            "add should remain when override disables optimizer"
        );
    }

    #[test]
    fn optimizer_override_can_enable_even_with_env_disable() {
        let _env = OptimizerEnvGuard::new();
        std::env::set_var("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");
        crate::set_rust_optimizer_override(true);
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c2 = b.constant_bit32(int, 2);
        let c3 = b.constant_bit32(int, 3);
        let add = b.i_add(int, None, c2, c3).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_wrapped_block(&words);
        crate::clear_rust_optimizer_override();
        std::env::remove_var("SPIRV_TOOLS_DISABLE_RUST_OPT");

        assert!(optimized.success, "override should still succeed");

        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized.words, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        for inst in module.all_inst_iter() {
            if inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id == Some(add)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(5)]
            {
                saw_const = true;
            }
            assert_ne!(
                inst.class.opcode,
                rspirv::spirv::Op::IAdd,
                "add should fold"
            );
        }
        assert!(
            saw_const,
            "override enable should run optimizer even when env disables it"
        );
    }

    #[test]
    fn optimizer_affine_gcd_add_folds_to_constant() {
        let mut b = Builder::new();
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c6 = b.constant_bit32(int, 6);
        let c12 = b.constant_bit32(int, 12);
        let x = b.constant_bit32(int, 4);
        let mul = b.i_mul(int, None, c6, x).expect("mul");
        let add = b.i_add(int, None, mul, c12).expect("add");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        let mut saw_ops = false;
        for inst in module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if inst.result_id == Some(add)
                        && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(36)]
                    {
                        saw_const = true;
                    }
                }
                Op::IMul | Op::IAdd if inst.result_id == Some(add) => saw_ops = true,
                _ => {}
            }
        }
        assert!(saw_const, "affine gcd add should fold to const 36");
        assert!(!saw_ops, "mul/add should be removed after folding");
    }

    #[test]
    fn optimizer_affine_gcd_sub_folds_to_constant() {
        let mut b = Builder::new();
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c14 = b.constant_bit32(int, 14);
        let c21 = b.constant_bit32(int, 21);
        let x = b.constant_bit32(int, 2);
        let mul = b.i_mul(int, None, c14, x).expect("mul");
        let sub = b.i_sub(int, None, mul, c21).expect("sub");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let module = loader.module();

        let mut saw_const = false;
        let mut saw_ops = false;
        for inst in module.all_inst_iter() {
            match inst.class.opcode {
                Op::Constant => {
                    if inst.result_id == Some(sub)
                        && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(7)]
                    {
                        saw_const = true;
                    }
                }
                Op::IMul | Op::ISub if inst.result_id == Some(sub) => saw_ops = true,
                _ => {}
            }
        }
        assert!(saw_const, "affine gcd sub should fold to const 7");
        assert!(!saw_ops, "mul/sub should be removed after folding");
    }

    #[test]
    fn optimizer_rewrites_umod_pow2_to_bitmask() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c5 = b.constant_bit32(int, 5);
        let c1 = b.constant_bit32(int, 1);
        let c8 = b.constant_bit32(int, 8);
        let x = b.i_add(int, None, c5, c1).expect("iadd");
        let umod = b.u_mod(int, None, x, c8).expect("umod");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut saw_band = false;
        let mut saw_const = false;
        for inst in optimized_module.all_inst_iter() {
            assert_ne!(inst.class.opcode, Op::UMod, "umod should be rewritten");
            if inst.class.opcode == Op::BitwiseAnd {
                saw_band = true;
                let mask_is_7 = inst
                    .operands
                    .iter()
                    .any(|op| matches!(op, rspirv::dr::Operand::LiteralBit32(7)));
                assert!(mask_is_7, "expected mask 7 for modulo by 8");
                assert_eq!(
                    inst.result_id,
                    Some(umod),
                    "should reuse original result id for bitmask"
                );
            }
            if inst.class.opcode == Op::Constant
                && inst.result_id == Some(umod)
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(6)]
            {
                saw_const = true;
            }
        }
        assert!(
            saw_band || saw_const,
            "expected bitwise mask or folded constant to replace umod"
        );
    }

    #[test]
    fn optimizer_folds_rem_by_one_to_zero() {
        let mut b = Builder::new();
        let void = b.type_void();
        let int = b.type_int(32, 0);
        let func_ty = b.type_function(void, vec![]);
        b.capability(rspirv::spirv::Capability::Shader);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );
        let _func = b
            .begin_function(void, None, FunctionControl::NONE, func_ty)
            .unwrap();
        let _ = b.begin_block(None).unwrap();
        let c5 = b.constant_bit32(int, 5);
        let c1 = b.constant_bit32(int, 1);
        let umod = b.u_mod(int, None, c5, c1).expect("umod");
        let srem = b.s_rem(int, None, c5, c1).expect("srem");
        b.ret().unwrap();
        b.end_function().unwrap();
        let words = b.module().assemble();

        let optimized = optimize_basic_block(&words).expect("optimizer runs");
        let mut loader = Loader::new();
        rspirv::binary::parse_words(&optimized, &mut loader).expect("parse optimized");
        let optimized_module = loader.module();

        let mut saw_zero_umod = false;
        let mut saw_zero_srem = false;
        for inst in optimized_module.all_inst_iter() {
            if inst.class.opcode == Op::UMod {
                panic!("umod by 1 should be folded");
            }
            if inst.class.opcode == Op::SRem {
                panic!("srem by 1 should be folded");
            }
            if inst.class.opcode == Op::Constant
                && inst.operands == vec![rspirv::dr::Operand::LiteralBit32(0)]
            {
                saw_zero_umod |= inst.result_id == Some(umod) || inst.result_id.is_some();
                saw_zero_srem |= inst.result_id == Some(srem) || inst.result_id.is_some();
            }
        }
        assert!(saw_zero_umod, "umod by 1 should fold to zero");
        assert!(saw_zero_srem, "srem by 1 should fold to zero");
    }
}
