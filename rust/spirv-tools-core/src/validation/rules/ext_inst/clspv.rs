//! NonSemantic.ClspvReflection extended instruction validation.
//!
//! This module validates CLSpv reflection extended instructions which
//! are used to convey kernel metadata from OpenCL kernels compiled with clspv.

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::ResultId;

// ============================================================================
// NonSemantic.ClspvReflection Instruction Type and Extension Trait
// ============================================================================

/// A NonSemantic.ClspvReflection instruction opcode.
///
/// This is a newtype wrapper around the raw u32 opcode value, providing
/// type safety and extension methods for CLSpv reflection instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClspvInstruction(pub u32);

#[allow(missing_docs)]
impl ClspvInstruction {
    // Version 1 instructions
    pub const KERNEL: Self = Self(1);
    pub const ARGUMENT_INFO: Self = Self(2);
    pub const ARGUMENT_STORAGE_BUFFER: Self = Self(3);
    pub const ARGUMENT_UNIFORM: Self = Self(4);
    pub const ARGUMENT_POD_STORAGE_BUFFER: Self = Self(5);
    pub const ARGUMENT_POD_UNIFORM: Self = Self(6);
    pub const ARGUMENT_POD_PUSH_CONSTANT: Self = Self(7);
    pub const ARGUMENT_SAMPLED_IMAGE: Self = Self(8);
    pub const ARGUMENT_STORAGE_IMAGE: Self = Self(9);
    pub const ARGUMENT_SAMPLER: Self = Self(10);
    pub const ARGUMENT_WORKGROUP: Self = Self(11);
    pub const SPEC_CONSTANT_WORKGROUP_SIZE: Self = Self(12);
    pub const SPEC_CONSTANT_GLOBAL_OFFSET: Self = Self(13);
    pub const SPEC_CONSTANT_WORK_DIM: Self = Self(14);
    pub const PUSH_CONSTANT_GLOBAL_OFFSET: Self = Self(15);
    pub const PUSH_CONSTANT_ENQUEUED_LOCAL_SIZE: Self = Self(16);
    pub const PUSH_CONSTANT_GLOBAL_SIZE: Self = Self(17);
    pub const PUSH_CONSTANT_REGION_OFFSET: Self = Self(18);
    pub const PUSH_CONSTANT_NUM_WORKGROUPS: Self = Self(19);
    pub const PUSH_CONSTANT_REGION_GROUP_OFFSET: Self = Self(20);
    pub const CONSTANT_DATA_STORAGE_BUFFER: Self = Self(21);
    pub const CONSTANT_DATA_UNIFORM: Self = Self(22);
    pub const LITERAL_SAMPLER: Self = Self(23);
    pub const PROPERTY_REQUIRED_WORKGROUP_SIZE: Self = Self(24);
    // Version 3 instructions
    pub const SPEC_CONSTANT_SUBGROUP_MAX_SIZE: Self = Self(25);
    // Version 4 instructions
    pub const ARGUMENT_POINTER_PUSH_CONSTANT: Self = Self(26);
    pub const ARGUMENT_POINTER_UNIFORM: Self = Self(27);
    pub const PROGRAM_SCOPE_VARIABLES_STORAGE_BUFFER: Self = Self(28);
    pub const PROGRAM_SCOPE_VARIABLE_POINTER_RELOCATION: Self = Self(29);
    pub const IMAGE_ARGUMENT_INFO_CHANNEL_ORDER_PUSH_CONSTANT: Self = Self(30);
    pub const IMAGE_ARGUMENT_INFO_CHANNEL_DATA_TYPE_PUSH_CONSTANT: Self = Self(31);
    pub const IMAGE_ARGUMENT_INFO_CHANNEL_ORDER_UNIFORM: Self = Self(32);
    pub const IMAGE_ARGUMENT_INFO_CHANNEL_DATA_TYPE_UNIFORM: Self = Self(33);
    // Version 5 instructions
    pub const ARGUMENT_STORAGE_TEXEL_BUFFER: Self = Self(34);
    pub const ARGUMENT_UNIFORM_TEXEL_BUFFER: Self = Self(35);
    pub const CONSTANT_DATA_POINTER_PUSH_CONSTANT: Self = Self(36);
    pub const PROGRAM_SCOPE_VARIABLE_POINTER_PUSH_CONSTANT: Self = Self(37);
    // Version 6 instructions
    pub const PRINTF_INFO: Self = Self(38);
    pub const PRINTF_BUFFER_STORAGE_BUFFER: Self = Self(39);
    pub const PRINTF_BUFFER_POINTER_PUSH_CONSTANT: Self = Self(40);
    // Version 7 instructions
    pub const NORMALIZED_SAMPLER_MASK_PUSH_CONSTANT: Self = Self(41);
    pub const WORKGROUP_VARIABLE_SIZE: Self = Self(42);
}

/// Extension trait for CLSpv reflection instruction metadata.
pub trait ClspvInstructionExt {
    /// Returns the name of this CLSpv reflection instruction.
    fn name(&self) -> &'static str;

    /// Returns the minimum CLSpv version required for this instruction.
    fn min_version(&self) -> u32;
}

impl ClspvInstructionExt for ClspvInstruction {
    fn name(&self) -> &'static str {
        match self.0 {
            1 => "Kernel",
            2 => "ArgumentInfo",
            3 => "ArgumentStorageBuffer",
            4 => "ArgumentUniform",
            5 => "ArgumentPodStorageBuffer",
            6 => "ArgumentPodUniform",
            7 => "ArgumentPodPushConstant",
            8 => "ArgumentSampledImage",
            9 => "ArgumentStorageImage",
            10 => "ArgumentSampler",
            11 => "ArgumentWorkgroup",
            12 => "SpecConstantWorkgroupSize",
            13 => "SpecConstantGlobalOffset",
            14 => "SpecConstantWorkDim",
            15 => "PushConstantGlobalOffset",
            16 => "PushConstantEnqueuedLocalSize",
            17 => "PushConstantGlobalSize",
            18 => "PushConstantRegionOffset",
            19 => "PushConstantNumWorkgroups",
            20 => "PushConstantRegionGroupOffset",
            21 => "ConstantDataStorageBuffer",
            22 => "ConstantDataUniform",
            23 => "LiteralSampler",
            24 => "PropertyRequiredWorkgroupSize",
            25 => "SpecConstantSubgroupMaxSize",
            26 => "ArgumentPointerPushConstant",
            27 => "ArgumentPointerUniform",
            28 => "ProgramScopeVariablesStorageBuffer",
            29 => "ProgramScopeVariablePointerRelocation",
            30 => "ImageArgumentInfoChannelOrderPushConstant",
            31 => "ImageArgumentInfoChannelDataTypePushConstant",
            32 => "ImageArgumentInfoChannelOrderUniform",
            33 => "ImageArgumentInfoChannelDataTypeUniform",
            34 => "ArgumentStorageTexelBuffer",
            35 => "ArgumentUniformTexelBuffer",
            36 => "ConstantDataPointerPushConstant",
            37 => "ProgramScopeVariablePointerPushConstant",
            38 => "PrintfInfo",
            39 => "PrintfBufferStorageBuffer",
            40 => "PrintfBufferPointerPushConstant",
            41 => "NormalizedSamplerMaskPushConstant",
            42 => "WorkgroupVariableSize",
            _ => "Unknown",
        }
    }

    fn min_version(&self) -> u32 {
        match self.0 {
            1..=24 => 1,
            25 => 3,
            26..=33 => 4,
            34..=37 => 5,
            38..=40 => 6,
            41..=42 => 7,
            _ => 1,
        }
    }
}

/// Checks if a result ID is a 32-bit unsigned integer constant.
fn is_uint32_constant(id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(id) {
        if let Some(inst) = ctx.definitions.get(&result_id) {
            if inst.class.opcode == Op::Constant {
                if let Some(type_id) = inst.result_type {
                    if let Ok(type_result_id) = ResultId::try_from(type_id) {
                        if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                            if type_inst.class.opcode == Op::TypeInt {
                                // Check it's 32-bit unsigned
                                if let (Some(Operand::LiteralBit32(width)), Some(Operand::LiteralBit32(signedness))) =
                                    (type_inst.operands.first(), type_inst.operands.get(1))
                                {
                                    return *width == 32 && *signedness == 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Finds the CLSpv reflection import ID and version in the module.
fn get_clspv_import(ctx: &ValidationContext<'_>) -> Option<(u32, u32)> {
    for inst in &ctx.module.ext_inst_imports {
        if inst.class.opcode == Op::ExtInstImport {
            if let Some(Operand::LiteralString(name)) = inst.operands.first() {
                // Parse "NonSemantic.ClspvReflection.N" format
                if let Some(rest) = name.strip_prefix("NonSemantic.ClspvReflection.") {
                    if let Ok(version) = rest.parse::<u32>() {
                        if let Some(id) = inst.result_id {
                            return Some((id, version));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Checks if an instruction is a CLSpv reflection extended instruction.
fn is_clspv_ext_inst(inst: &Instruction, import_id: u32) -> bool {
    if inst.class.opcode != Op::ExtInst {
        return false;
    }
    if let Some(Operand::IdRef(set_id)) = inst.operands.first() {
        return *set_id == import_id;
    }
    false
}

/// Gets the CLSpv instruction opcode from an OpExtInst instruction.
fn get_clspv_opcode(inst: &Instruction) -> Option<ClspvInstruction> {
    if inst.class.opcode != Op::ExtInst {
        return None;
    }
    if let Some(Operand::LiteralExtInstInteger(opcode)) = inst.operands.get(1) {
        return Some(ClspvInstruction(*opcode));
    }
    None
}

/// Checks if an instruction's result is a CLSpv Kernel instruction.
fn is_clspv_kernel(id: u32, import_id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(id) {
        if let Some(inst) = ctx.definitions.get(&result_id) {
            if is_clspv_ext_inst(inst, import_id) {
                if let Some(opcode) = get_clspv_opcode(inst) {
                    return opcode == ClspvInstruction::KERNEL;
                }
            }
        }
    }
    false
}

/// Checks if an instruction's result is a CLSpv ArgumentInfo instruction.
fn is_clspv_argument_info(id: u32, import_id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(id) {
        if let Some(inst) = ctx.definitions.get(&result_id) {
            if is_clspv_ext_inst(inst, import_id) {
                if let Some(opcode) = get_clspv_opcode(inst) {
                    return opcode == ClspvInstruction::ARGUMENT_INFO;
                }
            }
        }
    }
    false
}

/// Checks if a result ID is an OpString.
fn is_op_string(id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(id) {
        if let Some(inst) = ctx.definitions.get(&result_id) {
            return inst.class.opcode == Op::String;
        }
    }
    false
}

// ============================================================================
// CLSpv Reflection Validation Rule
// ============================================================================

/// Validates NonSemantic.ClspvReflection extended instructions.
///
/// This validates the CLSpv reflection metadata instructions used to describe
/// OpenCL C kernels compiled to SPIR-V for Vulkan.
///
/// Key validations:
/// - All instructions must return OpTypeVoid
/// - Version requirements for newer instructions
/// - Operand types (OpString, uint32 constants, Kernel/ArgumentInfo references)
pub struct ClspvReflectionRule;

impl ValidationRule for ClspvReflectionRule {
    fn name(&self) -> &'static str {
        "clspv-reflection"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, version) = match get_clspv_import(ctx) {
            Some(v) => v,
            None => return Ok(()), // No CLSpv import
        };

        // Iterate through all instructions in the module
        // CLSpv reflection instructions can appear in:
        // 1. Global ext_inst_imports section (already checked for import)
        // 2. Annotations section
        // 3. Types/Constants section (usually here)
        // 4. Function bodies

        // Check type and constant declarations
        for inst in ctx.module.types_global_values.iter() {
            if !is_clspv_ext_inst(inst, import_id) {
                continue;
            }
            self.validate_clspv_instruction(inst, import_id, version, ctx)?;
        }

        // Check function body instructions
        for function in &ctx.module.functions {
            for block in &function.blocks {
                for inst in &block.instructions {
                    if !is_clspv_ext_inst(inst, import_id) {
                        continue;
                    }
                    self.validate_clspv_instruction(inst, import_id, version, ctx)?;
                }
            }
        }

        Ok(())
    }
}

impl ClspvReflectionRule {
    /// Validates a single CLSpv reflection instruction.
    fn validate_clspv_instruction(
        &self,
        inst: &Instruction,
        import_id: u32,
        version: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        let opcode = match get_clspv_opcode(inst) {
            Some(op) => op,
            None => return Ok(()),
        };

        let inst_name = opcode.name();

        // All CLSpv reflection instructions must return void
        if let Some(result_type) = inst.result_type {
            if let Ok(result_type_id) = ResultId::try_from(result_type) {
                if let Some(type_inst) = ctx.definitions.get(&result_type_id) {
                    if type_inst.class.opcode != Op::TypeVoid {
                        return Err(ValidationError::ClspvResultTypeMustBeVoid {
                            instruction: inst_name,
                        }.into());
                    }
                }
            }
        }

        // Check version requirements
        let min_version = opcode.min_version();
        if version < min_version {
            return Err(ValidationError::ClspvVersionRequired {
                instruction: inst_name,
                required: min_version,
                found: version,
            }.into());
        }

        // Validate specific instruction operands
        match opcode {
            ClspvInstruction::KERNEL => {
                self.validate_kernel(inst, inst_name, version, ctx)?;
            }
            ClspvInstruction::ARGUMENT_INFO => {
                self.validate_argument_info(inst, inst_name, ctx)?;
            }
            ClspvInstruction::ARGUMENT_STORAGE_BUFFER
            | ClspvInstruction::ARGUMENT_UNIFORM
            | ClspvInstruction::ARGUMENT_SAMPLED_IMAGE
            | ClspvInstruction::ARGUMENT_STORAGE_IMAGE
            | ClspvInstruction::ARGUMENT_SAMPLER
            | ClspvInstruction::ARGUMENT_STORAGE_TEXEL_BUFFER
            | ClspvInstruction::ARGUMENT_UNIFORM_TEXEL_BUFFER => {
                self.validate_argument_buffer(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::ARGUMENT_POD_STORAGE_BUFFER
            | ClspvInstruction::ARGUMENT_POD_UNIFORM
            | ClspvInstruction::ARGUMENT_POINTER_UNIFORM => {
                self.validate_argument_offset_buffer(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::ARGUMENT_POD_PUSH_CONSTANT
            | ClspvInstruction::ARGUMENT_POINTER_PUSH_CONSTANT => {
                self.validate_argument_push_constant(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::ARGUMENT_WORKGROUP => {
                self.validate_argument_workgroup(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::SPEC_CONSTANT_WORKGROUP_SIZE
            | ClspvInstruction::SPEC_CONSTANT_GLOBAL_OFFSET => {
                self.validate_spec_constant_triple(inst, inst_name, ctx)?;
            }
            ClspvInstruction::SPEC_CONSTANT_WORK_DIM
            | ClspvInstruction::SPEC_CONSTANT_SUBGROUP_MAX_SIZE => {
                self.validate_spec_constant_single(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PUSH_CONSTANT_GLOBAL_OFFSET
            | ClspvInstruction::PUSH_CONSTANT_ENQUEUED_LOCAL_SIZE
            | ClspvInstruction::PUSH_CONSTANT_GLOBAL_SIZE
            | ClspvInstruction::PUSH_CONSTANT_REGION_OFFSET
            | ClspvInstruction::PUSH_CONSTANT_NUM_WORKGROUPS
            | ClspvInstruction::PUSH_CONSTANT_REGION_GROUP_OFFSET => {
                self.validate_push_constant_offset_size(inst, inst_name, ctx)?;
            }
            ClspvInstruction::CONSTANT_DATA_STORAGE_BUFFER
            | ClspvInstruction::CONSTANT_DATA_UNIFORM => {
                self.validate_constant_data(inst, inst_name, ctx)?;
            }
            ClspvInstruction::LITERAL_SAMPLER => {
                self.validate_literal_sampler(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PROPERTY_REQUIRED_WORKGROUP_SIZE => {
                self.validate_property_required_workgroup_size(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::PROGRAM_SCOPE_VARIABLES_STORAGE_BUFFER => {
                self.validate_program_scope_variables(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PROGRAM_SCOPE_VARIABLE_POINTER_RELOCATION => {
                self.validate_pointer_relocation(inst, inst_name, ctx)?;
            }
            ClspvInstruction::IMAGE_ARGUMENT_INFO_CHANNEL_ORDER_PUSH_CONSTANT
            | ClspvInstruction::IMAGE_ARGUMENT_INFO_CHANNEL_DATA_TYPE_PUSH_CONSTANT => {
                self.validate_image_metadata_push_constant(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::IMAGE_ARGUMENT_INFO_CHANNEL_ORDER_UNIFORM
            | ClspvInstruction::IMAGE_ARGUMENT_INFO_CHANNEL_DATA_TYPE_UNIFORM => {
                self.validate_image_metadata_uniform(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::CONSTANT_DATA_POINTER_PUSH_CONSTANT
            | ClspvInstruction::PROGRAM_SCOPE_VARIABLE_POINTER_PUSH_CONSTANT => {
                self.validate_push_constant_data(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PRINTF_INFO => {
                self.validate_printf_info(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PRINTF_BUFFER_STORAGE_BUFFER => {
                self.validate_printf_storage_buffer(inst, inst_name, ctx)?;
            }
            ClspvInstruction::PRINTF_BUFFER_POINTER_PUSH_CONSTANT => {
                self.validate_printf_push_constant(inst, inst_name, ctx)?;
            }
            ClspvInstruction::NORMALIZED_SAMPLER_MASK_PUSH_CONSTANT => {
                self.validate_normalized_sampler_mask(inst, inst_name, import_id, ctx)?;
            }
            ClspvInstruction::WORKGROUP_VARIABLE_SIZE => {
                self.validate_workgroup_variable_size(inst, inst_name, import_id, ctx)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Validates Kernel instruction.
    fn validate_kernel(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        version: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 4: function (IdRef to entry point)
        // Operand 5: name (OpString)
        // Operand 6 (optional, v5+): num_args (uint32 constant)
        // Operand 7 (optional, v5+): flags (uint32 constant)
        // Operand 8 (optional, v6+): attributes (OpString)

        if inst.operands.len() > 3 {
            if let Some(Operand::IdRef(name_id)) = inst.operands.get(3) {
                if !is_op_string(*name_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Name",
                    }.into());
                }
            }
        }

        // Version 5+ requires num_args
        if version >= 5 && inst.operands.len() > 4 {
            if let Some(Operand::IdRef(num_args_id)) = inst.operands.get(4) {
                if !is_uint32_constant(*num_args_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "NumArguments",
                    }.into());
                }
            }
        }

        // Version 5+ may have flags
        if version >= 5 && inst.operands.len() > 5 {
            if let Some(Operand::IdRef(flags_id)) = inst.operands.get(5) {
                if !is_uint32_constant(*flags_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "Flags",
                    }.into());
                }
            }
        }

        // Version 6+ may have attributes string
        if version >= 6 && inst.operands.len() > 6 {
            if let Some(Operand::IdRef(attrs_id)) = inst.operands.get(6) {
                if !is_op_string(*attrs_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Attributes",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ArgumentInfo instruction.
    fn validate_argument_info(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 4: name (OpString)
        // Operand 5 (optional): type_name (OpString)
        // Operand 6 (optional): address_qualifier (uint32 constant)
        // Operand 7 (optional): access_qualifier (uint32 constant)
        // Operand 8 (optional): type_qualifier (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(name_id)) = inst.operands.get(2) {
                if !is_op_string(*name_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Name",
                    }.into());
                }
            }
        }

        // Optional type_name
        if inst.operands.len() > 3 {
            if let Some(Operand::IdRef(type_name_id)) = inst.operands.get(3) {
                if !is_op_string(*type_name_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "TypeName",
                    }.into());
                }
            }
        }

        // Optional address qualifier
        if inst.operands.len() > 4 {
            if let Some(Operand::IdRef(addr_qual_id)) = inst.operands.get(4) {
                if !is_uint32_constant(*addr_qual_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "AddressQualifier",
                    }.into());
                }
            }
        }

        // Optional access qualifier
        if inst.operands.len() > 5 {
            if let Some(Operand::IdRef(access_qual_id)) = inst.operands.get(5) {
                if !is_uint32_constant(*access_qual_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "AccessQualifier",
                    }.into());
                }
            }
        }

        // Optional type qualifier
        if inst.operands.len() > 6 {
            if let Some(Operand::IdRef(type_qual_id)) = inst.operands.get(6) {
                if !is_uint32_constant(*type_qual_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "TypeQualifier",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ArgumentStorageBuffer, ArgumentUniform, etc.
    fn validate_argument_buffer(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: ordinal (uint32 constant)
        // Operand 4: descriptor_set (uint32 constant)
        // Operand 5: binding (uint32 constant)
        // Operand 6 (optional): arg_info (ArgumentInfo instruction)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        // Ordinal
        if inst.operands.len() > 3 {
            if let Some(Operand::IdRef(ordinal_id)) = inst.operands.get(3) {
                if !is_uint32_constant(*ordinal_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "Ordinal",
                    }.into());
                }
            }
        }

        // Descriptor set
        if inst.operands.len() > 4 {
            if let Some(Operand::IdRef(desc_set_id)) = inst.operands.get(4) {
                if !is_uint32_constant(*desc_set_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "DescriptorSet",
                    }.into());
                }
            }
        }

        // Binding
        if inst.operands.len() > 5 {
            if let Some(Operand::IdRef(binding_id)) = inst.operands.get(5) {
                if !is_uint32_constant(*binding_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "Binding",
                    }.into());
                }
            }
        }

        // Optional ArgumentInfo
        if inst.operands.len() > 6 {
            if let Some(Operand::IdRef(arg_info_id)) = inst.operands.get(6) {
                if !is_clspv_argument_info(*arg_info_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeArgumentInfo {
                        instruction: inst_name,
                        operand_name: "ArgumentInfo",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ArgumentPodStorageBuffer, ArgumentPodUniform, ArgumentPointerUniform.
    fn validate_argument_offset_buffer(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Same as argument_buffer but with offset and size
        // Operand 2: kernel
        // Operand 3: ordinal
        // Operand 4: descriptor_set
        // Operand 5: binding
        // Operand 6: offset
        // Operand 7: size
        // Operand 8 (optional): arg_info

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Ordinal"), (4, "DescriptorSet"), (5, "Binding"), (6, "Offset"), (7, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        // Optional ArgumentInfo
        if inst.operands.len() > 8 {
            if let Some(Operand::IdRef(arg_info_id)) = inst.operands.get(8) {
                if !is_clspv_argument_info(*arg_info_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeArgumentInfo {
                        instruction: inst_name,
                        operand_name: "ArgumentInfo",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ArgumentPodPushConstant, ArgumentPointerPushConstant.
    fn validate_argument_push_constant(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel
        // Operand 3: ordinal
        // Operand 4: offset
        // Operand 5: size
        // Operand 6 (optional): arg_info

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Ordinal"), (4, "Offset"), (5, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        // Optional ArgumentInfo
        if inst.operands.len() > 6 {
            if let Some(Operand::IdRef(arg_info_id)) = inst.operands.get(6) {
                if !is_clspv_argument_info(*arg_info_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeArgumentInfo {
                        instruction: inst_name,
                        operand_name: "ArgumentInfo",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ArgumentWorkgroup.
    fn validate_argument_workgroup(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel
        // Operand 3: ordinal
        // Operand 4: spec_id
        // Operand 5: elem_size
        // Operand 6 (optional): arg_info

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Ordinal"), (4, "SpecId"), (5, "ElemSize")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        // Optional ArgumentInfo
        if inst.operands.len() > 6 {
            if let Some(Operand::IdRef(arg_info_id)) = inst.operands.get(6) {
                if !is_clspv_argument_info(*arg_info_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeArgumentInfo {
                        instruction: inst_name,
                        operand_name: "ArgumentInfo",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates SpecConstantWorkgroupSize, SpecConstantGlobalOffset.
    fn validate_spec_constant_triple(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: x_id (uint32 constant)
        // Operand 3: y_id (uint32 constant)
        // Operand 4: z_id (uint32 constant)

        for (idx, name) in [(2, "X"), (3, "Y"), (4, "Z")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates SpecConstantWorkDim, SpecConstantSubgroupMaxSize.
    fn validate_spec_constant_single(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: id (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(id)) = inst.operands.get(2) {
                if !is_uint32_constant(*id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "SpecId",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates PushConstant* instructions.
    fn validate_push_constant_offset_size(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: offset (uint32 constant)
        // Operand 3: size (uint32 constant)

        for (idx, name) in [(2, "Offset"), (3, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates ConstantDataStorageBuffer, ConstantDataUniform.
    fn validate_constant_data(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: descriptor_set (uint32 constant)
        // Operand 3: binding (uint32 constant)
        // Operand 4: data (OpString)

        for (idx, name) in [(2, "DescriptorSet"), (3, "Binding")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        if inst.operands.len() > 4 {
            if let Some(Operand::IdRef(data_id)) = inst.operands.get(4) {
                if !is_op_string(*data_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Data",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates LiteralSampler.
    fn validate_literal_sampler(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: descriptor_set (uint32 constant)
        // Operand 3: binding (uint32 constant)
        // Operand 4: mask (uint32 constant)

        for (idx, name) in [(2, "DescriptorSet"), (3, "Binding"), (4, "Mask")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates PropertyRequiredWorkgroupSize.
    fn validate_property_required_workgroup_size(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: x (uint32 constant)
        // Operand 4: y (uint32 constant)
        // Operand 5: z (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "X"), (4, "Y"), (5, "Z")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates ProgramScopeVariablesStorageBuffer.
    fn validate_program_scope_variables(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: descriptor_set (uint32 constant)
        // Operand 3: binding (uint32 constant)
        // Operand 4: data (OpString)

        for (idx, name) in [(2, "DescriptorSet"), (3, "Binding")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        if inst.operands.len() > 4 {
            if let Some(Operand::IdRef(data_id)) = inst.operands.get(4) {
                if !is_op_string(*data_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Data",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates ProgramScopeVariablePointerRelocation.
    fn validate_pointer_relocation(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: object_offset (uint32 constant)
        // Operand 3: pointer_offset (uint32 constant)
        // Operand 4: pointer_size (uint32 constant)

        for (idx, name) in [(2, "ObjectOffset"), (3, "PointerOffset"), (4, "PointerSize")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates ImageArgumentInfoChannel*PushConstant.
    fn validate_image_metadata_push_constant(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: ordinal (uint32 constant)
        // Operand 4: offset (uint32 constant)
        // Operand 5: size (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Ordinal"), (4, "Offset"), (5, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates ImageArgumentInfoChannel*Uniform.
    fn validate_image_metadata_uniform(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: ordinal (uint32 constant)
        // Operand 4: descriptor_set (uint32 constant)
        // Operand 5: binding (uint32 constant)
        // Operand 6: offset (uint32 constant)
        // Operand 7: size (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [
            (3, "Ordinal"),
            (4, "DescriptorSet"),
            (5, "Binding"),
            (6, "Offset"),
            (7, "Size"),
        ] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates ConstantDataPointerPushConstant, ProgramScopeVariablePointerPushConstant.
    fn validate_push_constant_data(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: offset (uint32 constant)
        // Operand 3: size (uint32 constant)
        // Operand 4: data (OpString)

        for (idx, name) in [(2, "Offset"), (3, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        if inst.operands.len() > 4 {
            if let Some(Operand::IdRef(data_id)) = inst.operands.get(4) {
                if !is_op_string(*data_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "Data",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates PrintfInfo.
    fn validate_printf_info(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: printf_id (uint32 constant)
        // Operand 3: format_string (OpString)
        // Operand 4..N: arg_sizes (uint32 constants)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(printf_id)) = inst.operands.get(2) {
                if !is_uint32_constant(*printf_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "PrintfID",
                    }.into());
                }
            }
        }

        if inst.operands.len() > 3 {
            if let Some(Operand::IdRef(format_id)) = inst.operands.get(3) {
                if !is_op_string(*format_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeString {
                        instruction: inst_name,
                        operand_name: "FormatString",
                    }.into());
                }
            }
        }

        // Variable number of arg_sizes
        for i in 4..inst.operands.len() {
            if let Some(Operand::IdRef(arg_size_id)) = inst.operands.get(i) {
                if !is_uint32_constant(*arg_size_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                        instruction: inst_name,
                        operand_name: "ArgSize",
                    }.into());
                }
            }
        }

        Ok(())
    }

    /// Validates PrintfBufferStorageBuffer.
    fn validate_printf_storage_buffer(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: descriptor_set (uint32 constant)
        // Operand 3: binding (uint32 constant)
        // Operand 4: size (uint32 constant)

        for (idx, name) in [(2, "DescriptorSet"), (3, "Binding"), (4, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates PrintfBufferPointerPushConstant.
    fn validate_printf_push_constant(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: offset (uint32 constant)
        // Operand 3: size (uint32 constant)
        // Operand 4: buffer_size (uint32 constant)

        for (idx, name) in [(2, "Offset"), (3, "Size"), (4, "BufferSize")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates NormalizedSamplerMaskPushConstant.
    fn validate_normalized_sampler_mask(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: ordinal (uint32 constant)
        // Operand 4: offset (uint32 constant)
        // Operand 5: size (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Ordinal"), (4, "Offset"), (5, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates WorkgroupVariableSize.
    fn validate_workgroup_variable_size(
        &self,
        inst: &Instruction,
        inst_name: &'static str,
        import_id: u32,
        ctx: &ValidationContext<'_>,
    ) -> ValidationResult {
        // Operand 2: kernel (Kernel instruction)
        // Operand 3: argument (uint32 constant - ordinal)
        // Operand 4: size (uint32 constant)

        if inst.operands.len() > 2 {
            if let Some(Operand::IdRef(kernel_id)) = inst.operands.get(2) {
                if !is_clspv_kernel(*kernel_id, import_id, ctx) {
                    return Err(ValidationError::ClspvOperandMustBeKernel {
                        instruction: inst_name,
                        operand_name: "Kernel",
                    }.into());
                }
            }
        }

        for (idx, name) in [(3, "Argument"), (4, "Size")] {
            if inst.operands.len() > idx {
                if let Some(Operand::IdRef(id)) = inst.operands.get(idx) {
                    if !is_uint32_constant(*id, ctx) {
                        return Err(ValidationError::ClspvOperandMustBeUint32Constant {
                            instruction: inst_name,
                            operand_name: name,
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clspv_name_lookup() {
        assert_eq!(ClspvInstruction::KERNEL.name(), "Kernel");
        assert_eq!(ClspvInstruction::ARGUMENT_INFO.name(), "ArgumentInfo");
        assert_eq!(ClspvInstruction::ARGUMENT_STORAGE_BUFFER.name(), "ArgumentStorageBuffer");
        assert_eq!(ClspvInstruction::PRINTF_INFO.name(), "PrintfInfo");
        assert_eq!(ClspvInstruction::WORKGROUP_VARIABLE_SIZE.name(), "WorkgroupVariableSize");
        assert_eq!(ClspvInstruction(999).name(), "Unknown");
    }

    #[test]
    fn test_clspv_min_version() {
        // Version 1 instructions
        assert_eq!(ClspvInstruction::KERNEL.min_version(), 1);
        assert_eq!(ClspvInstruction::ARGUMENT_INFO.min_version(), 1);
        assert_eq!(ClspvInstruction::PROPERTY_REQUIRED_WORKGROUP_SIZE.min_version(), 1);

        // Version 3 instruction
        assert_eq!(ClspvInstruction::SPEC_CONSTANT_SUBGROUP_MAX_SIZE.min_version(), 3);

        // Version 4 instructions
        assert_eq!(ClspvInstruction::ARGUMENT_POINTER_PUSH_CONSTANT.min_version(), 4);
        assert_eq!(ClspvInstruction::IMAGE_ARGUMENT_INFO_CHANNEL_DATA_TYPE_UNIFORM.min_version(), 4);

        // Version 5 instructions
        assert_eq!(ClspvInstruction::ARGUMENT_STORAGE_TEXEL_BUFFER.min_version(), 5);
        assert_eq!(ClspvInstruction::PROGRAM_SCOPE_VARIABLE_POINTER_PUSH_CONSTANT.min_version(), 5);

        // Version 6 instructions
        assert_eq!(ClspvInstruction::PRINTF_INFO.min_version(), 6);
        assert_eq!(ClspvInstruction::PRINTF_BUFFER_POINTER_PUSH_CONSTANT.min_version(), 6);

        // Version 7 instructions
        assert_eq!(ClspvInstruction::NORMALIZED_SAMPLER_MASK_PUSH_CONSTANT.min_version(), 7);
        assert_eq!(ClspvInstruction::WORKGROUP_VARIABLE_SIZE.min_version(), 7);
    }
}
