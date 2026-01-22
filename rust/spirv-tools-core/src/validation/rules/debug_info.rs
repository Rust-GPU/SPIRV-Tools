//! Debug Info extended instruction validation rules.
//!
//! This module validates SPIR-V debug info extended instructions including:
//!
//! - OpenCL.DebugInfo.100 extended instruction set
//! - NonSemantic.Shader.DebugInfo.100 extended instruction set
//!
//! Both instruction sets share a common core set of debug instructions
//! (CommonDebugInfo) with slight encoding differences.

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::ResultId;
use crate::validation::ValidationResult;

// ============================================================================
// Common Debug Info Instruction Constants
// ============================================================================

/// Common debug info instruction opcodes shared between OpenCL.DebugInfo.100
/// and NonSemantic.Shader.DebugInfo.100.
#[allow(dead_code)]
mod debug_info {
    pub const DEBUG_INFO_NONE: u32 = 0;
    pub const DEBUG_COMPILATION_UNIT: u32 = 1;
    pub const DEBUG_TYPE_BASIC: u32 = 2;
    pub const DEBUG_TYPE_POINTER: u32 = 3;
    pub const DEBUG_TYPE_QUALIFIER: u32 = 4;
    pub const DEBUG_TYPE_ARRAY: u32 = 5;
    pub const DEBUG_TYPE_VECTOR: u32 = 6;
    pub const DEBUG_TYPEDEF: u32 = 7;
    pub const DEBUG_TYPE_FUNCTION: u32 = 8;
    pub const DEBUG_TYPE_ENUM: u32 = 9;
    pub const DEBUG_TYPE_COMPOSITE: u32 = 10;
    pub const DEBUG_TYPE_MEMBER: u32 = 11;
    pub const DEBUG_TYPE_INHERITANCE: u32 = 12;
    pub const DEBUG_TYPE_PTR_TO_MEMBER: u32 = 13;
    pub const DEBUG_TYPE_TEMPLATE: u32 = 14;
    pub const DEBUG_TYPE_TEMPLATE_PARAMETER: u32 = 15;
    pub const DEBUG_TYPE_TEMPLATE_TEMPLATE_PARAMETER: u32 = 16;
    pub const DEBUG_TYPE_TEMPLATE_PARAMETER_PACK: u32 = 17;
    pub const DEBUG_GLOBAL_VARIABLE: u32 = 18;
    pub const DEBUG_FUNCTION_DECLARATION: u32 = 19;
    pub const DEBUG_FUNCTION: u32 = 20;
    pub const DEBUG_LEXICAL_BLOCK: u32 = 21;
    pub const DEBUG_LEXICAL_BLOCK_DISCRIMINATOR: u32 = 22;
    pub const DEBUG_SCOPE: u32 = 23;
    pub const DEBUG_NO_SCOPE: u32 = 24;
    pub const DEBUG_INLINED_AT: u32 = 25;
    pub const DEBUG_LOCAL_VARIABLE: u32 = 26;
    pub const DEBUG_INLINED_VARIABLE: u32 = 27;
    pub const DEBUG_DECLARE: u32 = 28;
    pub const DEBUG_VALUE: u32 = 29;
    pub const DEBUG_OPERATION: u32 = 30;
    pub const DEBUG_EXPRESSION: u32 = 31;
    pub const DEBUG_MACRO_DEF: u32 = 32;
    pub const DEBUG_MACRO_UNDEF: u32 = 33;
    pub const DEBUG_IMPORTED_ENTITY: u32 = 34;
    pub const DEBUG_SOURCE: u32 = 35;
}

/// Debug info extension import names.
const OPENCL_DEBUG_INFO_100: &str = "OpenCL.DebugInfo.100";
const NONSEMANTIC_SHADER_DEBUG_INFO_100: &str = "NonSemantic.Shader.DebugInfo.100";

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the debug info import ID from the module.
/// Returns (import_id, is_vulkan_debug_info) where is_vulkan_debug_info indicates
/// NonSemantic.Shader.DebugInfo.100 (uses constants instead of literals).
fn get_debug_info_import(ctx: &ValidationContext<'_>) -> Option<(u32, bool)> {
    for inst in &ctx.module.ext_inst_imports {
        if inst.class.opcode == Op::ExtInstImport {
            if let Some(Operand::LiteralString(name)) = inst.operands.first() {
                if name == OPENCL_DEBUG_INFO_100 {
                    return inst.result_id.map(|id| (id, false));
                }
                if name == NONSEMANTIC_SHADER_DEBUG_INFO_100 {
                    return inst.result_id.map(|id| (id, true));
                }
            }
        }
    }
    None
}

/// Check if an instruction is a debug info extended instruction.
fn is_debug_info_ext_inst(inst: &Instruction, import_id: u32) -> bool {
    if inst.class.opcode != Op::ExtInst {
        return false;
    }
    // Operand 0 is the extension set ID
    if let Some(Operand::IdRef(ext_set)) = inst.operands.first() {
        return *ext_set == import_id;
    }
    false
}

/// Get the debug info opcode from an OpExtInst instruction.
fn get_debug_info_opcode(inst: &Instruction) -> Option<u32> {
    // Operand 1 is the instruction number (LiteralExtInstInteger)
    if let Some(Operand::LiteralExtInstInteger(opcode)) = inst.operands.get(1) {
        return Some(*opcode);
    }
    None
}

/// Get the name of a debug info instruction.
fn get_debug_info_name(opcode: u32) -> &'static str {
    match opcode {
        debug_info::DEBUG_INFO_NONE => "DebugInfoNone",
        debug_info::DEBUG_COMPILATION_UNIT => "DebugCompilationUnit",
        debug_info::DEBUG_TYPE_BASIC => "DebugTypeBasic",
        debug_info::DEBUG_TYPE_POINTER => "DebugTypePointer",
        debug_info::DEBUG_TYPE_QUALIFIER => "DebugTypeQualifier",
        debug_info::DEBUG_TYPE_ARRAY => "DebugTypeArray",
        debug_info::DEBUG_TYPE_VECTOR => "DebugTypeVector",
        debug_info::DEBUG_TYPEDEF => "DebugTypedef",
        debug_info::DEBUG_TYPE_FUNCTION => "DebugTypeFunction",
        debug_info::DEBUG_TYPE_ENUM => "DebugTypeEnum",
        debug_info::DEBUG_TYPE_COMPOSITE => "DebugTypeComposite",
        debug_info::DEBUG_TYPE_MEMBER => "DebugTypeMember",
        debug_info::DEBUG_TYPE_INHERITANCE => "DebugTypeInheritance",
        debug_info::DEBUG_TYPE_PTR_TO_MEMBER => "DebugTypePtrToMember",
        debug_info::DEBUG_TYPE_TEMPLATE => "DebugTypeTemplate",
        debug_info::DEBUG_TYPE_TEMPLATE_PARAMETER => "DebugTypeTemplateParameter",
        debug_info::DEBUG_TYPE_TEMPLATE_TEMPLATE_PARAMETER => "DebugTypeTemplateTemplateParameter",
        debug_info::DEBUG_TYPE_TEMPLATE_PARAMETER_PACK => "DebugTypeTemplateParameterPack",
        debug_info::DEBUG_GLOBAL_VARIABLE => "DebugGlobalVariable",
        debug_info::DEBUG_FUNCTION_DECLARATION => "DebugFunctionDeclaration",
        debug_info::DEBUG_FUNCTION => "DebugFunction",
        debug_info::DEBUG_LEXICAL_BLOCK => "DebugLexicalBlock",
        debug_info::DEBUG_LEXICAL_BLOCK_DISCRIMINATOR => "DebugLexicalBlockDiscriminator",
        debug_info::DEBUG_SCOPE => "DebugScope",
        debug_info::DEBUG_NO_SCOPE => "DebugNoScope",
        debug_info::DEBUG_INLINED_AT => "DebugInlinedAt",
        debug_info::DEBUG_LOCAL_VARIABLE => "DebugLocalVariable",
        debug_info::DEBUG_INLINED_VARIABLE => "DebugInlinedVariable",
        debug_info::DEBUG_DECLARE => "DebugDeclare",
        debug_info::DEBUG_VALUE => "DebugValue",
        debug_info::DEBUG_OPERATION => "DebugOperation",
        debug_info::DEBUG_EXPRESSION => "DebugExpression",
        debug_info::DEBUG_MACRO_DEF => "DebugMacroDef",
        debug_info::DEBUG_MACRO_UNDEF => "DebugMacroUndef",
        debug_info::DEBUG_IMPORTED_ENTITY => "DebugImportedEntity",
        debug_info::DEBUG_SOURCE => "DebugSource",
        _ => "Unknown",
    }
}

/// Check if an operand is a constant (OpConstant).
fn is_constant(operand_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if let Ok(result_id) = ResultId::try_from(operand_id) {
        if let Some(inst) = definitions.get(&result_id) {
            return matches!(
                inst.class.opcode,
                Op::Constant
                    | Op::ConstantTrue
                    | Op::ConstantFalse
                    | Op::ConstantNull
                    | Op::SpecConstant
                    | Op::SpecConstantTrue
                    | Op::SpecConstantFalse
            );
        }
    }
    false
}

/// Check if an operand is an OpString.
fn is_op_string(operand_id: u32, definitions: &HashMap<ResultId, Instruction>) -> bool {
    if let Ok(result_id) = ResultId::try_from(operand_id) {
        if let Some(inst) = definitions.get(&result_id) {
            return inst.class.opcode == Op::String;
        }
    }
    false
}

/// Check if an operand is a debug info instruction with a specific opcode.
fn is_debug_info_instruction(
    operand_id: u32,
    expected_opcode: u32,
    import_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    if let Ok(result_id) = ResultId::try_from(operand_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if is_debug_info_ext_inst(inst, import_id) {
                if let Some(opcode) = get_debug_info_opcode(inst) {
                    return opcode == expected_opcode;
                }
            }
        }
    }
    false
}

/// Check if an operand is a debug type instruction.
fn is_debug_type(
    operand_id: u32,
    import_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
    allow_template_param: bool,
) -> bool {
    if let Ok(result_id) = ResultId::try_from(operand_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if is_debug_info_ext_inst(inst, import_id) {
                if let Some(opcode) = get_debug_info_opcode(inst) {
                    // Debug type opcodes are 2-14 (DebugTypeBasic to DebugTypeTemplate)
                    if (debug_info::DEBUG_TYPE_BASIC..=debug_info::DEBUG_TYPE_TEMPLATE)
                        .contains(&opcode)
                    {
                        return true;
                    }
                    // Optionally allow template parameters
                    if allow_template_param
                        && (opcode == debug_info::DEBUG_TYPE_TEMPLATE_PARAMETER
                            || opcode == debug_info::DEBUG_TYPE_TEMPLATE_TEMPLATE_PARAMETER)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if an operand is a lexical scope (DebugCompilationUnit, DebugFunction,
/// DebugLexicalBlock, or DebugTypeComposite).
fn is_lexical_scope(
    operand_id: u32,
    import_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    if let Ok(result_id) = ResultId::try_from(operand_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if is_debug_info_ext_inst(inst, import_id) {
                if let Some(opcode) = get_debug_info_opcode(inst) {
                    return opcode == debug_info::DEBUG_COMPILATION_UNIT
                        || opcode == debug_info::DEBUG_FUNCTION
                        || opcode == debug_info::DEBUG_LEXICAL_BLOCK
                        || opcode == debug_info::DEBUG_TYPE_COMPOSITE;
                }
            }
        }
    }
    false
}

/// Get an operand ID at a specific index (after ext set and opcode).
fn get_operand_id(inst: &Instruction, index: usize) -> Option<u32> {
    // Operands start at index 2 (after ext set ID and instruction number)
    inst.operands.get(index).and_then(|op| match op {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    })
}

/// Get a literal operand at a specific index (for OpenCL.DebugInfo.100).
fn get_literal_operand(inst: &Instruction, index: usize) -> Option<u32> {
    inst.operands.get(index).and_then(|op| match op {
        Operand::LiteralBit32(val) => Some(*val),
        _ => None,
    })
}

// ============================================================================
// Debug Info Source Rule
// ============================================================================

/// Validates DebugSource instructions.
///
/// DebugSource operands:
/// - File: must be OpString
/// - Text (optional): must be OpString if present
pub struct DebugSourceRule;

impl ValidationRule for DebugSourceRule {
    fn name(&self) -> &'static str {
        "debug-source"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_SOURCE {
                continue;
            }

            // Operand 2 (index 2): File - must be OpString
            if let Some(file_id) = get_operand_id(inst, 2) {
                if !is_op_string(file_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotString {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "File",
                    }
                    .into());
                }
            }

            // Operand 3 (index 3): Text (optional) - must be OpString if present
            if let Some(text_id) = get_operand_id(inst, 3) {
                if is_vulkan {
                    // In Vulkan debug info, this is an ID reference to a constant or string
                    if !is_op_string(text_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotString {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Text",
                        }
                        .into());
                    }
                } else if !is_op_string(text_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotString {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Text",
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Compilation Unit Rule
// ============================================================================

/// Validates DebugCompilationUnit instructions.
///
/// DebugCompilationUnit operands:
/// - Version: must be constant uint
/// - DWARF Version: must be constant uint
/// - Source: must be DebugSource
/// - Language: must be constant uint
pub struct DebugCompilationUnitRule;

impl ValidationRule for DebugCompilationUnitRule {
    fn name(&self) -> &'static str {
        "debug-compilation-unit"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_COMPILATION_UNIT {
                continue;
            }

            // In Vulkan debug info, all operands are constant IDs
            // In OpenCL debug info, some operands are literals
            if is_vulkan {
                // Operand 2: Version - must be constant
                if let Some(version_id) = get_operand_id(inst, 2) {
                    if !is_constant(version_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Version",
                        }
                        .into());
                    }
                }

                // Operand 3: DWARF Version - must be constant
                if let Some(dwarf_id) = get_operand_id(inst, 3) {
                    if !is_constant(dwarf_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "DWARF Version",
                        }
                        .into());
                    }
                }

                // Operand 4: Source - must be DebugSource
                if let Some(source_id) = get_operand_id(inst, 4) {
                    if !is_debug_info_instruction(
                        source_id,
                        debug_info::DEBUG_SOURCE,
                        import_id,
                        ctx.definitions,
                    ) {
                        return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Source",
                            expected: "DebugSource",
                        }
                        .into());
                    }
                }

                // Operand 5: Language - must be constant
                if let Some(lang_id) = get_operand_id(inst, 5) {
                    if !is_constant(lang_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Language",
                        }
                        .into());
                    }
                }
            } else {
                // OpenCL.DebugInfo.100 uses literals for Version, DWARF Version, Language
                // and ID for Source

                // Operand 4 (index 4 after skipping result/type): Source - must be DebugSource
                if let Some(source_id) = get_operand_id(inst, 4) {
                    if !is_debug_info_instruction(
                        source_id,
                        debug_info::DEBUG_SOURCE,
                        import_id,
                        ctx.definitions,
                    ) {
                        return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Source",
                            expected: "DebugSource",
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Type Basic Rule
// ============================================================================

/// Validates DebugTypeBasic instructions.
///
/// DebugTypeBasic operands:
/// - Name: must be OpString
/// - Size: must be OpConstant
/// - Encoding: must be constant uint
pub struct DebugTypeBasicRule;

impl ValidationRule for DebugTypeBasicRule {
    fn name(&self) -> &'static str {
        "debug-type-basic"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_TYPE_BASIC {
                continue;
            }

            // Operand 2: Name - must be OpString
            if let Some(name_id) = get_operand_id(inst, 2) {
                if !is_op_string(name_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotString {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Name",
                    }
                    .into());
                }
            }

            // Operand 3: Size - must be OpConstant
            if let Some(size_id) = get_operand_id(inst, 3) {
                if !is_constant(size_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotConstant {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Size",
                    }
                    .into());
                }
            }

            // Operand 4: Encoding - must be constant (Vulkan) or literal (OpenCL)
            if is_vulkan {
                if let Some(encoding_id) = get_operand_id(inst, 4) {
                    if !is_constant(encoding_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Encoding",
                        }
                        .into());
                    }
                }
            }
            // For OpenCL, encoding is a literal - nothing to validate
        }

        Ok(())
    }
}

// ============================================================================
// Debug Type Vector Rule
// ============================================================================

/// Validates DebugTypeVector instructions.
///
/// DebugTypeVector operands:
/// - BaseType: must be DebugTypeBasic
/// - ComponentCount: must be constant 1-4
pub struct DebugTypeVectorRule;

impl ValidationRule for DebugTypeVectorRule {
    fn name(&self) -> &'static str {
        "debug-type-vector"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_TYPE_VECTOR {
                continue;
            }

            // Operand 2: BaseType - must be DebugTypeBasic
            if let Some(base_type_id) = get_operand_id(inst, 2) {
                if !is_debug_info_instruction(
                    base_type_id,
                    debug_info::DEBUG_TYPE_BASIC,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Base Type",
                        expected: "DebugTypeBasic",
                    }
                    .into());
                }
            }

            // Operand 3: ComponentCount - must be 1-4
            if is_vulkan {
                // In Vulkan debug info, this is a constant ID
                if let Some(count_id) = get_operand_id(inst, 3) {
                    if !is_constant(count_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Component Count",
                        }
                        .into());
                    }
                    // Check the constant value is 1-4
                    if let Ok(result_id) = ResultId::try_from(count_id) {
                        if let Some(const_inst) = ctx.definitions.get(&result_id) {
                            if let Some(Operand::LiteralBit32(val)) = const_inst.operands.first() {
                                if *val == 0 || *val > 4 {
                                    return Err(
                                        ValidationError::DebugTypeVectorInvalidComponentCount {
                                            count: *val,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                    }
                }
            } else {
                // In OpenCL, this is a literal
                if let Some(count) = get_literal_operand(inst, 3) {
                    if count == 0 || count > 4 {
                        return Err(ValidationError::DebugTypeVectorInvalidComponentCount {
                            count,
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Type Pointer Rule
// ============================================================================

/// Validates DebugTypePointer instructions.
///
/// DebugTypePointer operands:
/// - BaseType: must be a debug type
/// - StorageClass: must be constant uint
/// - Flags: must be constant uint
pub struct DebugTypePointerRule;

impl ValidationRule for DebugTypePointerRule {
    fn name(&self) -> &'static str {
        "debug-type-pointer"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_TYPE_POINTER {
                continue;
            }

            // Operand 2: BaseType - must be a debug type
            if let Some(base_type_id) = get_operand_id(inst, 2) {
                if !is_debug_type(base_type_id, import_id, ctx.definitions, false) {
                    return Err(ValidationError::DebugInfoOperandNotDebugType {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Base Type",
                    }
                    .into());
                }
            }

            // Operand 3: StorageClass - must be constant (Vulkan)
            if is_vulkan {
                if let Some(sc_id) = get_operand_id(inst, 3) {
                    if !is_constant(sc_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Storage Class",
                        }
                        .into());
                    }
                }
            }

            // Operand 4: Flags - must be constant (Vulkan)
            if is_vulkan {
                if let Some(flags_id) = get_operand_id(inst, 4) {
                    if !is_constant(flags_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Flags",
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Local Variable Rule
// ============================================================================

/// Validates DebugLocalVariable instructions.
///
/// DebugLocalVariable operands:
/// - Name: must be OpString
/// - Type: must be a debug type
/// - Source: must be DebugSource
/// - Line: must be constant uint
/// - Column: must be constant uint
/// - Scope: must be a lexical scope
/// - Flags: must be constant uint
/// - ArgNumber (optional): must be constant uint
pub struct DebugLocalVariableRule;

impl ValidationRule for DebugLocalVariableRule {
    fn name(&self) -> &'static str {
        "debug-local-variable"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_LOCAL_VARIABLE {
                continue;
            }

            // Operand 2: Name - must be OpString
            if let Some(name_id) = get_operand_id(inst, 2) {
                if !is_op_string(name_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotString {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Name",
                    }
                    .into());
                }
            }

            // Operand 3: Type - must be a debug type
            if let Some(type_id) = get_operand_id(inst, 3) {
                if !is_debug_type(type_id, import_id, ctx.definitions, false) {
                    return Err(ValidationError::DebugInfoOperandNotDebugType {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Type",
                    }
                    .into());
                }
            }

            // Operand 4: Source - must be DebugSource
            if let Some(source_id) = get_operand_id(inst, 4) {
                if !is_debug_info_instruction(
                    source_id,
                    debug_info::DEBUG_SOURCE,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Source",
                        expected: "DebugSource",
                    }
                    .into());
                }
            }

            if is_vulkan {
                // Operand 5: Line - must be constant
                if let Some(line_id) = get_operand_id(inst, 5) {
                    if !is_constant(line_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Line",
                        }
                        .into());
                    }
                }

                // Operand 6: Column - must be constant
                if let Some(col_id) = get_operand_id(inst, 6) {
                    if !is_constant(col_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Column",
                        }
                        .into());
                    }
                }
            }

            // Scope operand - must be a lexical scope
            let scope_index = if is_vulkan { 7 } else { 7 };
            if let Some(scope_id) = get_operand_id(inst, scope_index) {
                if !is_lexical_scope(scope_id, import_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotLexicalScope {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Scope",
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Scope Rule
// ============================================================================

/// Validates DebugScope instructions.
///
/// DebugScope operands:
/// - Scope: must be a lexical scope
/// - InlinedAt (optional): must be DebugInlinedAt
pub struct DebugScopeRule;

impl ValidationRule for DebugScopeRule {
    fn name(&self) -> &'static str {
        "debug-scope"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, _is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_SCOPE {
                continue;
            }

            // Operand 2: Scope - must be a lexical scope
            if let Some(scope_id) = get_operand_id(inst, 2) {
                if !is_lexical_scope(scope_id, import_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotLexicalScope {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Scope",
                    }
                    .into());
                }
            }

            // Operand 3 (optional): InlinedAt - must be DebugInlinedAt
            if let Some(inlined_at_id) = get_operand_id(inst, 3) {
                if !is_debug_info_instruction(
                    inlined_at_id,
                    debug_info::DEBUG_INLINED_AT,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "InlinedAt",
                        expected: "DebugInlinedAt",
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Declare Rule
// ============================================================================

/// Validates DebugDeclare instructions.
///
/// DebugDeclare operands:
/// - LocalVariable: must be DebugLocalVariable
/// - Variable: must be an ID
/// - Expression: must be DebugExpression
pub struct DebugDeclareRule;

impl ValidationRule for DebugDeclareRule {
    fn name(&self) -> &'static str {
        "debug-declare"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, _is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_DECLARE {
                continue;
            }

            // Operand 2: LocalVariable - must be DebugLocalVariable
            if let Some(local_var_id) = get_operand_id(inst, 2) {
                if !is_debug_info_instruction(
                    local_var_id,
                    debug_info::DEBUG_LOCAL_VARIABLE,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Local Variable",
                        expected: "DebugLocalVariable",
                    }
                    .into());
                }
            }

            // Operand 4: Expression - must be DebugExpression
            if let Some(expr_id) = get_operand_id(inst, 4) {
                if !is_debug_info_instruction(
                    expr_id,
                    debug_info::DEBUG_EXPRESSION,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Expression",
                        expected: "DebugExpression",
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Value Rule
// ============================================================================

/// Validates DebugValue instructions.
///
/// DebugValue operands:
/// - LocalVariable: must be DebugLocalVariable
/// - Value: must be an ID
/// - Expression: must be DebugExpression
/// - Indexes (optional): must be constant uints
pub struct DebugValueRule;

impl ValidationRule for DebugValueRule {
    fn name(&self) -> &'static str {
        "debug-value"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_VALUE {
                continue;
            }

            // Operand 2: LocalVariable - must be DebugLocalVariable
            if let Some(local_var_id) = get_operand_id(inst, 2) {
                if !is_debug_info_instruction(
                    local_var_id,
                    debug_info::DEBUG_LOCAL_VARIABLE,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Local Variable",
                        expected: "DebugLocalVariable",
                    }
                    .into());
                }
            }

            // Operand 4: Expression - must be DebugExpression
            if let Some(expr_id) = get_operand_id(inst, 4) {
                if !is_debug_info_instruction(
                    expr_id,
                    debug_info::DEBUG_EXPRESSION,
                    import_id,
                    ctx.definitions,
                ) {
                    return Err(ValidationError::DebugInfoOperandNotDebugInstruction {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Expression",
                        expected: "DebugExpression",
                    }
                    .into());
                }
            }

            // Remaining operands are optional Indexes - must be constants in Vulkan
            if is_vulkan {
                for i in 5..inst.operands.len() {
                    if let Some(index_id) = get_operand_id(inst, i) {
                        if !is_constant(index_id, ctx.definitions) {
                            return Err(ValidationError::DebugInfoOperandNotConstant {
                                instruction: get_debug_info_name(opcode),
                                operand_name: "Index",
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Debug Operation Rule
// ============================================================================

/// Validates DebugOperation instructions.
///
/// For Vulkan debug info, all operands must be constants.
pub struct DebugOperationRule;

impl ValidationRule for DebugOperationRule {
    fn name(&self) -> &'static str {
        "debug-operation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let (import_id, is_vulkan) = match get_debug_info_import(ctx) {
            Some((id, vulkan)) => (id, vulkan),
            None => return Ok(()), // No debug info import
        };

        // Only validate for Vulkan debug info
        if !is_vulkan {
            return Ok(());
        }

        for inst in ctx.module.all_inst_iter() {
            if !is_debug_info_ext_inst(inst, import_id) {
                continue;
            }

            let opcode = match get_debug_info_opcode(inst) {
                Some(op) => op,
                None => continue,
            };

            if opcode != debug_info::DEBUG_OPERATION {
                continue;
            }

            // Operand 2: Operation - must be constant
            if let Some(op_id) = get_operand_id(inst, 2) {
                if !is_constant(op_id, ctx.definitions) {
                    return Err(ValidationError::DebugInfoOperandNotConstant {
                        instruction: get_debug_info_name(opcode),
                        operand_name: "Operation",
                    }
                    .into());
                }
            }

            // Remaining operands are optional - must be constants
            for i in 3..inst.operands.len() {
                if let Some(operand_id) = get_operand_id(inst, i) {
                    if !is_constant(operand_id, ctx.definitions) {
                        return Err(ValidationError::DebugInfoOperandNotConstant {
                            instruction: get_debug_info_name(opcode),
                            operand_name: "Operand",
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All Debug Info Rules
// ============================================================================

/// Returns all debug info extended instruction validation rules.
pub fn all_debug_info_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(DebugSourceRule),
        Box::new(DebugCompilationUnitRule),
        Box::new(DebugTypeBasicRule),
        Box::new(DebugTypeVectorRule),
        Box::new(DebugTypePointerRule),
        Box::new(DebugLocalVariableRule),
        Box::new(DebugScopeRule),
        Box::new(DebugDeclareRule),
        Box::new(DebugValueRule),
        Box::new(DebugOperationRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::context::{TestContextData, ValidationRule};
    use rspirv::dr::Instruction;
    use rspirv::spirv::Op;

    #[test]
    fn test_debug_info_instruction_names() {
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_INFO_NONE),
            "DebugInfoNone"
        );
        assert_eq!(get_debug_info_name(debug_info::DEBUG_SOURCE), "DebugSource");
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_COMPILATION_UNIT),
            "DebugCompilationUnit"
        );
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_TYPE_BASIC),
            "DebugTypeBasic"
        );
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_TYPE_VECTOR),
            "DebugTypeVector"
        );
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_LOCAL_VARIABLE),
            "DebugLocalVariable"
        );
        assert_eq!(
            get_debug_info_name(debug_info::DEBUG_DECLARE),
            "DebugDeclare"
        );
        assert_eq!(get_debug_info_name(debug_info::DEBUG_VALUE), "DebugValue");
        assert_eq!(get_debug_info_name(999), "Unknown");
    }

    #[test]
    fn test_debug_info_constants() {
        assert_eq!(debug_info::DEBUG_INFO_NONE, 0);
        assert_eq!(debug_info::DEBUG_COMPILATION_UNIT, 1);
        assert_eq!(debug_info::DEBUG_TYPE_BASIC, 2);
        assert_eq!(debug_info::DEBUG_SOURCE, 35);
    }

    /// Helper to create an OpExtInstImport for debug info
    fn make_debug_info_import(id: u32, name: &str) -> Instruction {
        Instruction::new(
            Op::ExtInstImport,
            None,
            Some(id),
            vec![Operand::LiteralString(name.to_string())],
        )
    }

    /// Helper to create an OpString instruction
    fn make_op_string(id: u32, value: &str) -> Instruction {
        Instruction::new(
            Op::String,
            None,
            Some(id),
            vec![Operand::LiteralString(value.to_string())],
        )
    }

    /// Helper to create an OpConstant instruction
    fn make_op_constant(id: u32, type_id: u32, value: u32) -> Instruction {
        Instruction::new(
            Op::Constant,
            Some(type_id),
            Some(id),
            vec![Operand::LiteralBit32(value)],
        )
    }

    /// Helper to create a debug info extended instruction (for Vulkan NonSemantic)
    fn make_debug_ext_inst(
        result_id: u32,
        ext_set_id: u32,
        opcode: u32,
        operands: Vec<u32>,
    ) -> Instruction {
        let mut ops = vec![
            Operand::IdRef(ext_set_id),
            Operand::LiteralExtInstInteger(opcode),
        ];
        for op in operands {
            ops.push(Operand::IdRef(op));
        }
        Instruction::new(
            Op::ExtInst,
            Some(1), // void type
            Some(result_id),
            ops,
        )
    }

    /// Helper to create an OpTypeInt instruction
    fn make_type_int(id: u32, width: u32, signed: u32) -> Instruction {
        Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![Operand::LiteralBit32(width), Operand::LiteralBit32(signed)],
        )
    }

    // ========================================================================
    // Tests for no debug info import (should pass)
    // ========================================================================

    #[test]
    fn test_no_debug_info_import_passes() {
        let data = TestContextData::default();
        let ctx = data.as_context();

        assert!(DebugSourceRule.validate(&ctx).is_ok());
        assert!(DebugCompilationUnitRule.validate(&ctx).is_ok());
        assert!(DebugTypeBasicRule.validate(&ctx).is_ok());
        assert!(DebugTypeVectorRule.validate(&ctx).is_ok());
        assert!(DebugTypePointerRule.validate(&ctx).is_ok());
        assert!(DebugLocalVariableRule.validate(&ctx).is_ok());
        assert!(DebugScopeRule.validate(&ctx).is_ok());
        assert!(DebugDeclareRule.validate(&ctx).is_ok());
        assert!(DebugValueRule.validate(&ctx).is_ok());
        assert!(DebugOperationRule.validate(&ctx).is_ok());
    }

    // ========================================================================
    // DebugSource tests
    // ========================================================================

    #[test]
    fn test_debug_source_valid_file_only() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create OpString for file
        let file_string = make_op_string(2, "test.glsl");
        data.module.debug_string_source.push(file_string.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), file_string);

        // Create DebugSource with File operand
        let debug_source = make_debug_ext_inst(
            3,
            1, // ext set id
            debug_info::DEBUG_SOURCE,
            vec![2], // file string id
        );
        data.module.types_global_values.push(debug_source);

        let ctx = data.as_context();
        assert!(DebugSourceRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_debug_source_invalid_file_not_string() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create type int instead of OpString
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create DebugSource with non-string file operand
        let debug_source = make_debug_ext_inst(
            3,
            1, // ext set id
            debug_info::DEBUG_SOURCE,
            vec![2], // NOT a string
        );
        data.module.types_global_values.push(debug_source);

        let ctx = data.as_context();
        let result = DebugSourceRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugInfoOperandNotString {
                instruction,
                operand_name,
            } = spanned.error
            {
                assert_eq!(instruction, "DebugSource");
                assert_eq!(operand_name, "File");
            } else {
                panic!("Expected DebugInfoOperandNotString error");
            }
        }
    }

    // ========================================================================
    // DebugCompilationUnit tests
    // ========================================================================

    #[test]
    fn test_debug_compilation_unit_valid() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create constants
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        let version = make_op_constant(3, 2, 1);
        data.module.types_global_values.push(version.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), version);

        let dwarf_version = make_op_constant(4, 2, 4);
        data.module.types_global_values.push(dwarf_version.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), dwarf_version);

        let language = make_op_constant(6, 2, 2); // GLSL
        data.module.types_global_values.push(language.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), language);

        // Create DebugSource first
        let file_string = make_op_string(7, "test.glsl");
        data.module.debug_string_source.push(file_string.clone());
        data.definitions
            .insert(ResultId::try_from(7).unwrap(), file_string);

        let debug_source = make_debug_ext_inst(5, 1, debug_info::DEBUG_SOURCE, vec![7]);
        data.module.types_global_values.push(debug_source.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), debug_source);

        // Create DebugCompilationUnit
        let debug_cu = make_debug_ext_inst(
            8,
            1,
            debug_info::DEBUG_COMPILATION_UNIT,
            vec![3, 4, 5, 6], // version, dwarf_version, source, language
        );
        data.module.types_global_values.push(debug_cu);

        let ctx = data.as_context();
        assert!(DebugCompilationUnitRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_debug_compilation_unit_invalid_source() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create constants
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        let version = make_op_constant(3, 2, 1);
        data.module.types_global_values.push(version.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), version);

        let dwarf_version = make_op_constant(4, 2, 4);
        data.module.types_global_values.push(dwarf_version.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), dwarf_version);

        let language = make_op_constant(6, 2, 2);
        data.module.types_global_values.push(language.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), language);

        // Create NOT a DebugSource (just a constant)
        let fake_source = make_op_constant(5, 2, 0);
        data.module.types_global_values.push(fake_source.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), fake_source);

        // Create DebugCompilationUnit with invalid source
        let debug_cu = make_debug_ext_inst(
            8,
            1,
            debug_info::DEBUG_COMPILATION_UNIT,
            vec![3, 4, 5, 6], // source (5) is not a DebugSource
        );
        data.module.types_global_values.push(debug_cu);

        let ctx = data.as_context();
        let result = DebugCompilationUnitRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugInfoOperandNotDebugInstruction {
                instruction,
                operand_name,
                expected,
            } = spanned.error
            {
                assert_eq!(instruction, "DebugCompilationUnit");
                assert_eq!(operand_name, "Source");
                assert_eq!(expected, "DebugSource");
            } else {
                panic!("Expected DebugInfoOperandNotDebugInstruction error");
            }
        }
    }

    // ========================================================================
    // DebugTypeVector tests
    // ========================================================================

    #[test]
    fn test_debug_type_vector_valid_component_count() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create base type
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create OpString for name
        let name_string = make_op_string(3, "float");
        data.module.debug_string_source.push(name_string.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), name_string);

        // Create size constant
        let size_const = make_op_constant(4, 2, 32);
        data.module.types_global_values.push(size_const.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), size_const);

        // Create encoding constant
        let encoding_const = make_op_constant(5, 2, 4); // Float encoding
        data.module.types_global_values.push(encoding_const.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), encoding_const);

        // Create DebugTypeBasic first
        let debug_type_basic = make_debug_ext_inst(
            6,
            1,
            debug_info::DEBUG_TYPE_BASIC,
            vec![3, 4, 5], // name, size, encoding
        );
        data.module
            .types_global_values
            .push(debug_type_basic.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), debug_type_basic);

        // Create component count constant (valid: 4)
        let count_const = make_op_constant(7, 2, 4);
        data.module.types_global_values.push(count_const.clone());
        data.definitions
            .insert(ResultId::try_from(7).unwrap(), count_const);

        // Create DebugTypeVector
        let debug_type_vec = make_debug_ext_inst(
            8,
            1,
            debug_info::DEBUG_TYPE_VECTOR,
            vec![6, 7], // base type, component count
        );
        data.module.types_global_values.push(debug_type_vec);

        let ctx = data.as_context();
        assert!(DebugTypeVectorRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_debug_type_vector_invalid_component_count_zero() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create base type
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create OpString for name
        let name_string = make_op_string(3, "float");
        data.module.debug_string_source.push(name_string.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), name_string);

        // Create size and encoding constants
        let size_const = make_op_constant(4, 2, 32);
        data.module.types_global_values.push(size_const.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), size_const);

        let encoding_const = make_op_constant(5, 2, 4);
        data.module.types_global_values.push(encoding_const.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), encoding_const);

        // Create DebugTypeBasic
        let debug_type_basic =
            make_debug_ext_inst(6, 1, debug_info::DEBUG_TYPE_BASIC, vec![3, 4, 5]);
        data.module
            .types_global_values
            .push(debug_type_basic.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), debug_type_basic);

        // Create component count constant (INVALID: 0)
        let count_const = make_op_constant(7, 2, 0);
        data.module.types_global_values.push(count_const.clone());
        data.definitions
            .insert(ResultId::try_from(7).unwrap(), count_const);

        // Create DebugTypeVector
        let debug_type_vec = make_debug_ext_inst(8, 1, debug_info::DEBUG_TYPE_VECTOR, vec![6, 7]);
        data.module.types_global_values.push(debug_type_vec);

        let ctx = data.as_context();
        let result = DebugTypeVectorRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugTypeVectorInvalidComponentCount { count } = spanned.error {
                assert_eq!(count, 0);
            } else {
                panic!("Expected DebugTypeVectorInvalidComponentCount error");
            }
        }
    }

    #[test]
    fn test_debug_type_vector_invalid_component_count_too_large() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create base type
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create name string
        let name_string = make_op_string(3, "float");
        data.module.debug_string_source.push(name_string.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), name_string);

        // Create size and encoding constants
        let size_const = make_op_constant(4, 2, 32);
        data.module.types_global_values.push(size_const.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), size_const);

        let encoding_const = make_op_constant(5, 2, 4);
        data.module.types_global_values.push(encoding_const.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), encoding_const);

        // Create DebugTypeBasic
        let debug_type_basic =
            make_debug_ext_inst(6, 1, debug_info::DEBUG_TYPE_BASIC, vec![3, 4, 5]);
        data.module
            .types_global_values
            .push(debug_type_basic.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), debug_type_basic);

        // Create component count constant (INVALID: 5, exceeds max of 4)
        let count_const = make_op_constant(7, 2, 5);
        data.module.types_global_values.push(count_const.clone());
        data.definitions
            .insert(ResultId::try_from(7).unwrap(), count_const);

        // Create DebugTypeVector
        let debug_type_vec = make_debug_ext_inst(8, 1, debug_info::DEBUG_TYPE_VECTOR, vec![6, 7]);
        data.module.types_global_values.push(debug_type_vec);

        let ctx = data.as_context();
        let result = DebugTypeVectorRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugTypeVectorInvalidComponentCount { count } = spanned.error {
                assert_eq!(count, 5);
            } else {
                panic!("Expected DebugTypeVectorInvalidComponentCount error");
            }
        }
    }

    #[test]
    fn test_debug_type_vector_invalid_base_type() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create base type
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create a constant that is NOT DebugTypeBasic
        let fake_base = make_op_constant(6, 2, 0);
        data.module.types_global_values.push(fake_base.clone());
        data.definitions
            .insert(ResultId::try_from(6).unwrap(), fake_base);

        // Create component count constant
        let count_const = make_op_constant(7, 2, 4);
        data.module.types_global_values.push(count_const.clone());
        data.definitions
            .insert(ResultId::try_from(7).unwrap(), count_const);

        // Create DebugTypeVector with invalid base type
        let debug_type_vec = make_debug_ext_inst(
            8,
            1,
            debug_info::DEBUG_TYPE_VECTOR,
            vec![6, 7], // base type (6) is not DebugTypeBasic
        );
        data.module.types_global_values.push(debug_type_vec);

        let ctx = data.as_context();
        let result = DebugTypeVectorRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugInfoOperandNotDebugInstruction {
                instruction,
                operand_name,
                expected,
            } = spanned.error
            {
                assert_eq!(instruction, "DebugTypeVector");
                assert_eq!(operand_name, "Base Type");
                assert_eq!(expected, "DebugTypeBasic");
            } else {
                panic!("Expected DebugInfoOperandNotDebugInstruction error");
            }
        }
    }

    // ========================================================================
    // DebugTypeBasic tests
    // ========================================================================

    #[test]
    fn test_debug_type_basic_valid() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create int type for constants
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create OpString for name
        let name_string = make_op_string(3, "float");
        data.module.debug_string_source.push(name_string.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), name_string);

        // Create size constant
        let size_const = make_op_constant(4, 2, 32);
        data.module.types_global_values.push(size_const.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), size_const);

        // Create encoding constant
        let encoding_const = make_op_constant(5, 2, 4);
        data.module.types_global_values.push(encoding_const.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), encoding_const);

        // Create DebugTypeBasic
        let debug_type_basic = make_debug_ext_inst(
            6,
            1,
            debug_info::DEBUG_TYPE_BASIC,
            vec![3, 4, 5], // name, size, encoding
        );
        data.module.types_global_values.push(debug_type_basic);

        let ctx = data.as_context();
        assert!(DebugTypeBasicRule.validate(&ctx).is_ok());
    }

    #[test]
    fn test_debug_type_basic_invalid_name_not_string() {
        let mut data = TestContextData::default();

        // Create debug info import
        let import = make_debug_info_import(1, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create int type
        let type_int = make_type_int(2, 32, 0);
        data.module.types_global_values.push(type_int.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), type_int);

        // Create a constant instead of OpString for name (INVALID)
        let fake_name = make_op_constant(3, 2, 0);
        data.module.types_global_values.push(fake_name.clone());
        data.definitions
            .insert(ResultId::try_from(3).unwrap(), fake_name);

        // Create size constant
        let size_const = make_op_constant(4, 2, 32);
        data.module.types_global_values.push(size_const.clone());
        data.definitions
            .insert(ResultId::try_from(4).unwrap(), size_const);

        // Create encoding constant
        let encoding_const = make_op_constant(5, 2, 4);
        data.module.types_global_values.push(encoding_const.clone());
        data.definitions
            .insert(ResultId::try_from(5).unwrap(), encoding_const);

        // Create DebugTypeBasic with invalid name
        let debug_type_basic = make_debug_ext_inst(
            6,
            1,
            debug_info::DEBUG_TYPE_BASIC,
            vec![3, 4, 5], // name (3) is not OpString
        );
        data.module.types_global_values.push(debug_type_basic);

        let ctx = data.as_context();
        let result = DebugTypeBasicRule.validate(&ctx);
        assert!(result.is_err());
        if let Err(spanned) = result {
            if let ValidationError::DebugInfoOperandNotString {
                instruction,
                operand_name,
            } = spanned.error
            {
                assert_eq!(instruction, "DebugTypeBasic");
                assert_eq!(operand_name, "Name");
            } else {
                panic!("Expected DebugInfoOperandNotString error");
            }
        }
    }

    // ========================================================================
    // OpenCL.DebugInfo.100 tests
    // ========================================================================

    #[test]
    fn test_opencl_debug_info_import_recognized() {
        let mut data = TestContextData::default();

        // Create OpenCL debug info import
        let import = make_debug_info_import(1, OPENCL_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Create OpString for file
        let file_string = make_op_string(2, "test.cl");
        data.module.debug_string_source.push(file_string.clone());
        data.definitions
            .insert(ResultId::try_from(2).unwrap(), file_string);

        // Create DebugSource
        let debug_source = make_debug_ext_inst(3, 1, debug_info::DEBUG_SOURCE, vec![2]);
        data.module.types_global_values.push(debug_source);

        let ctx = data.as_context();
        assert!(DebugSourceRule.validate(&ctx).is_ok());
    }

    // ========================================================================
    // all_debug_info_rules tests
    // ========================================================================

    #[test]
    fn test_all_debug_info_rules_returns_expected_count() {
        let rules = all_debug_info_rules();
        assert_eq!(rules.len(), 10);
    }

    #[test]
    fn test_all_debug_info_rules_names() {
        let rules = all_debug_info_rules();
        let names: Vec<&str> = rules.iter().map(|r| r.name()).collect();

        assert!(names.contains(&"debug-source"));
        assert!(names.contains(&"debug-compilation-unit"));
        assert!(names.contains(&"debug-type-basic"));
        assert!(names.contains(&"debug-type-vector"));
        assert!(names.contains(&"debug-type-pointer"));
        assert!(names.contains(&"debug-local-variable"));
        assert!(names.contains(&"debug-scope"));
        assert!(names.contains(&"debug-declare"));
        assert!(names.contains(&"debug-value"));
        assert!(names.contains(&"debug-operation"));
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_is_debug_info_ext_inst() {
        let import_id = 10;

        // Valid debug info ext inst
        let valid_inst = Instruction::new(
            Op::ExtInst,
            Some(1),
            Some(100),
            vec![
                Operand::IdRef(import_id),
                Operand::LiteralExtInstInteger(debug_info::DEBUG_SOURCE),
            ],
        );
        assert!(is_debug_info_ext_inst(&valid_inst, import_id));

        // Wrong import id
        assert!(!is_debug_info_ext_inst(&valid_inst, 99));

        // Not an ExtInst
        let non_ext_inst = Instruction::new(Op::Nop, None, None, vec![]);
        assert!(!is_debug_info_ext_inst(&non_ext_inst, import_id));
    }

    #[test]
    fn test_get_debug_info_opcode() {
        let inst = Instruction::new(
            Op::ExtInst,
            Some(1),
            Some(100),
            vec![
                Operand::IdRef(10),
                Operand::LiteralExtInstInteger(debug_info::DEBUG_TYPE_BASIC),
            ],
        );
        assert_eq!(
            get_debug_info_opcode(&inst),
            Some(debug_info::DEBUG_TYPE_BASIC)
        );

        // No opcode operand
        let inst_no_opcode =
            Instruction::new(Op::ExtInst, Some(1), Some(100), vec![Operand::IdRef(10)]);
        assert_eq!(get_debug_info_opcode(&inst_no_opcode), None);
    }

    #[test]
    fn test_is_debug_type_range() {
        let mut data = TestContextData::default();

        let import_id: u32 = 1;

        // Create debug info import
        let import = make_debug_info_import(import_id, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Test various debug type opcodes (2-14)
        for opcode in debug_info::DEBUG_TYPE_BASIC..=debug_info::DEBUG_TYPE_TEMPLATE {
            let debug_type = make_debug_ext_inst(100 + opcode, import_id, opcode, vec![]);
            data.definitions
                .insert(ResultId::try_from(100 + opcode).unwrap(), debug_type);
            assert!(
                is_debug_type(100 + opcode, import_id, &data.definitions, false),
                "Opcode {} should be a debug type",
                opcode
            );
        }

        // Non-debug-type opcodes should not be considered debug types
        let non_type = make_debug_ext_inst(200, import_id, debug_info::DEBUG_SOURCE, vec![]);
        data.definitions
            .insert(ResultId::try_from(200).unwrap(), non_type);
        assert!(!is_debug_type(200, import_id, &data.definitions, false));
    }

    #[test]
    fn test_is_lexical_scope() {
        let mut data = TestContextData::default();

        let import_id: u32 = 1;

        // Create debug info import
        let import = make_debug_info_import(import_id, NONSEMANTIC_SHADER_DEBUG_INFO_100);
        data.module.ext_inst_imports.push(import);

        // Lexical scopes: DebugCompilationUnit, DebugFunction, DebugLexicalBlock, DebugTypeComposite
        let scopes = [
            (10, debug_info::DEBUG_COMPILATION_UNIT),
            (11, debug_info::DEBUG_FUNCTION),
            (12, debug_info::DEBUG_LEXICAL_BLOCK),
            (13, debug_info::DEBUG_TYPE_COMPOSITE),
        ];

        for (id, opcode) in scopes {
            let inst = make_debug_ext_inst(id, import_id, opcode, vec![]);
            data.definitions
                .insert(ResultId::try_from(id).unwrap(), inst);
            assert!(
                is_lexical_scope(id, import_id, &data.definitions),
                "Opcode {} should be a lexical scope",
                opcode
            );
        }

        // Non-lexical-scope should not be considered
        let non_scope = make_debug_ext_inst(20, import_id, debug_info::DEBUG_SOURCE, vec![]);
        data.definitions
            .insert(ResultId::try_from(20).unwrap(), non_scope);
        assert!(!is_lexical_scope(20, import_id, &data.definitions));
    }
}
