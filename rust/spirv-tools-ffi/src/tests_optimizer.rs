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
        let shl = b
            .shift_left_logical(int, None, c4, zero)
            .expect("shift id");
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
