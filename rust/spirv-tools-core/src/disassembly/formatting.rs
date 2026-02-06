use rspirv::binary::Disassemble;
use rspirv::dr::{Instruction, Operand};
use rspirv::spirv;
use std::collections::HashSet;
use std::num::FpCategory;

use super::names::{extract_id_ref, literal_operand_bits};
use super::types::*;
use crate::string_literal::render_string_literal;

pub(super) fn disassemble_with_format(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    type_table: &TypeTable,
    ext_inst_table: &ExtInstTable,
    value_types: &ValueTypeTable,
) -> String {
    let operands = if instruction.class.opcode == spirv::Op::Switch {
        format_switch_operands(instruction, literal_format, value_types)
    } else {
        format_ext_inst_operands(instruction, literal_format, ext_inst_table)
            .or_else(|| format_constant_operands(instruction, literal_format, type_table))
            .unwrap_or_else(|| disassemble_operands(&instruction.operands, literal_format))
    };
    let mut line = String::new();
    if let Some(result_id) = instruction.result_id {
        line.push('%');
        line.push_str(&result_id.to_string());
        line.push_str(" = ");
    }
    line.push_str("Op");
    line.push_str(instruction.class.opname);
    if let Some(result_type) = instruction.result_type {
        line.push(' ');
        line.push('%');
        line.push_str(&result_type.to_string());
    }
    if !operands.is_empty() {
        line.push(' ');
        line.push_str(&operands);
    }
    line
}

pub(super) fn disassemble_operands(operands: &[Operand], literal_format: LiteralFormat) -> String {
    operands
        .iter()
        .map(|operand| format_operand(operand, literal_format))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn format_constant_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    type_table: &TypeTable,
) -> Option<String> {
    if literal_format == LiteralFormat::Hexadecimal {
        return None;
    }
    match instruction.class.opcode {
        spirv::Op::Constant | spirv::Op::SpecConstant => {}
        _ => return None,
    }
    let type_id = instruction.result_type?;
    let literal = instruction.operands.first()?;
    let type_info = type_table.get(type_id)?;
    match type_info {
        TypeInfo::Int { width, signed } => format_integer_literal(literal, *width, *signed),
        TypeInfo::Float { width } => format_float_literal(literal, *width),
    }
}

pub(super) fn format_switch_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    value_types: &ValueTypeTable,
) -> String {
    if instruction.operands.len() < 2 {
        return disassemble_operands(&instruction.operands, literal_format);
    }
    let selector = &instruction.operands[0];
    let default_label = &instruction.operands[1];
    let selector_text = format_operand(selector, literal_format);
    let default_text = format_operand(default_label, literal_format);
    let selector_type = extract_id_ref(selector)
        .and_then(|id| value_types.get(id))
        .copied();
    let mut parts = vec![selector_text, default_text];
    let mut index = 2;
    while index + 1 < instruction.operands.len() {
        let literal = &instruction.operands[index];
        let label = &instruction.operands[index + 1];
        let literal_text = if literal_format == LiteralFormat::Hexadecimal {
            format_operand(literal, literal_format)
        } else if let Some(type_info) = selector_type {
            format_integer_from_type(literal, type_info)
                .unwrap_or_else(|| format_operand(literal, literal_format))
        } else {
            format_operand(literal, literal_format)
        };
        parts.push(literal_text);
        parts.push(format_operand(label, literal_format));
        index += 2;
    }
    parts.join(" ")
}

pub(super) fn format_integer_from_type(operand: &Operand, info: TypeInfo) -> Option<String> {
    match info {
        TypeInfo::Int { width, signed } => {
            literal_operand_bits(operand).map(|bits| format_integer_bits(bits, width, signed))
        }
        _ => None,
    }
}

pub(super) fn format_ext_inst_operands(
    instruction: &Instruction,
    literal_format: LiteralFormat,
    ext_inst_table: &ExtInstTable,
) -> Option<String> {
    if instruction.class.opcode != spirv::Op::ExtInst {
        return None;
    }
    if instruction.operands.len() < 2 {
        return None;
    }
    let mut parts = Vec::with_capacity(instruction.operands.len());
    let set_operand = &instruction.operands[0];
    parts.push(format_operand(set_operand, literal_format));
    let opcode_operand = &instruction.operands[1];
    let opcode_text = match (extract_id_ref(set_operand), opcode_operand) {
        (Some(set_id), Operand::LiteralExtInstInteger(value)) => {
            if let Some(name) = ext_inst_table.lookup_name(set_id, *value) {
                name.to_string()
            } else {
                format_operand(opcode_operand, literal_format)
            }
        }
        _ => format_operand(opcode_operand, literal_format),
    };
    parts.push(opcode_text);
    for operand in instruction.operands.iter().skip(2) {
        parts.push(format_operand(operand, literal_format));
    }
    Some(parts.join(" "))
}

pub(super) fn format_integer_literal(
    operand: &Operand,
    width: u32,
    signed: bool,
) -> Option<String> {
    let bits = match operand {
        Operand::LiteralBit32(value) => u64::from(*value),
        Operand::LiteralBit64(value) => *value,
        _ => return None,
    };
    Some(format_integer_bits(bits, width, signed))
}

pub(super) fn format_integer_bits(bits: u64, width: u32, signed: bool) -> String {
    if signed {
        if width >= 64 {
            (bits as i64).to_string()
        } else {
            let shift = 64 - width;
            let value = ((bits << shift) as i64) >> shift;
            value.to_string()
        }
    } else if width >= 64 {
        bits.to_string()
    } else {
        let mask = (1u64 << width) - 1;
        (bits & mask).to_string()
    }
}

pub(super) fn format_float_literal(operand: &Operand, width: u32) -> Option<String> {
    match width {
        16 => {
            let bits = match operand {
                Operand::LiteralBit32(value) => u64::from(*value & 0xffff),
                _ => return None,
            };
            Some(format_hex_float(bits, &HEX_FLOAT_F16))
        }
        32 => {
            let bits = match operand {
                Operand::LiteralBit32(value) => *value,
                _ => return None,
            };
            Some(format_f32_literal(bits))
        }
        64 => {
            let bits = match operand {
                Operand::LiteralBit64(value) => *value,
                _ => return None,
            };
            Some(format_f64_literal(bits))
        }
        _ => None,
    }
}

pub(super) fn format_f32_literal(bits: u32) -> String {
    let value = f32::from_bits(bits);
    match value.classify() {
        FpCategory::Zero | FpCategory::Normal => {
            format_decimal_float(f64::from(value), F32_DECIMAL_DIGITS)
        }
        _ => format_hex_float(u64::from(bits), &HEX_FLOAT_F32),
    }
}

pub(super) fn format_f64_literal(bits: u64) -> String {
    let value = f64::from_bits(bits);
    match value.classify() {
        FpCategory::Zero | FpCategory::Normal => format_decimal_float(value, F64_DECIMAL_DIGITS),
        _ => format_hex_float(bits, &HEX_FLOAT_F64),
    }
}

pub(super) fn format_decimal_float(value: f64, digits: usize) -> String {
    // Use pure Rust formatting - format with specified precision then trim trailing zeros
    // This mimics C's %g format specifier behavior
    let formatted = format!("{:.prec$}", value, prec = digits);

    // If it contains a decimal point, trim trailing zeros (like %g does)
    if formatted.contains('.') {
        let trimmed = formatted.trim_end_matches('0');
        // Don't leave a trailing decimal point
        let trimmed = trimmed.trim_end_matches('.');
        trimmed.to_string()
    } else {
        formatted
    }
}

pub(super) struct HexFloatConfig {
    pub(super) total_bits: u32,
    pub(super) exponent_bits: u32,
    pub(super) fraction_bits: u32,
    pub(super) exponent_bias: i32,
}

impl HexFloatConfig {
    pub(super) const fn fraction_nibbles(&self) -> u32 {
        self.fraction_bits.div_ceil(4)
    }

    pub(super) const fn overflow_bits(&self) -> u32 {
        self.fraction_nibbles() * 4 - self.fraction_bits
    }

    pub(super) const fn fraction_mask(&self) -> u64 {
        if self.fraction_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << self.fraction_bits) - 1
        }
    }
}

pub(super) const HEX_FLOAT_F16: HexFloatConfig = HexFloatConfig {
    total_bits: 16,
    exponent_bits: 5,
    fraction_bits: 10,
    exponent_bias: 15,
};
pub(super) const HEX_FLOAT_F32: HexFloatConfig = HexFloatConfig {
    total_bits: 32,
    exponent_bits: 8,
    fraction_bits: 23,
    exponent_bias: 127,
};
pub(super) const HEX_FLOAT_F64: HexFloatConfig = HexFloatConfig {
    total_bits: 64,
    exponent_bits: 11,
    fraction_bits: 52,
    exponent_bias: 1023,
};
pub(super) const F32_DECIMAL_DIGITS: usize = 9;
pub(super) const F64_DECIMAL_DIGITS: usize = 17;

pub(super) fn format_hex_float(bits: u64, config: &HexFloatConfig) -> String {
    let sign_mask = 1u64 << (config.total_bits - 1);
    let exponent_mask = if config.exponent_bits >= 64 {
        u64::MAX
    } else {
        ((1u64 << config.exponent_bits) - 1) << config.fraction_bits
    };
    let fraction_mask = config.fraction_mask();
    let sign = (bits & sign_mask) != 0;
    let raw_exponent = (bits & exponent_mask) >> config.fraction_bits;
    let mut fraction = (bits & fraction_mask) << config.overflow_bits();
    let is_zero = raw_exponent == 0 && fraction == 0;
    let is_denorm = raw_exponent == 0 && !is_zero;
    let mut exponent = if is_zero {
        0
    } else {
        raw_exponent as i64 - config.exponent_bias as i64
    };
    if is_denorm && config.fraction_bits + config.overflow_bits() > 0 {
        let top_bit = 1u64 << (config.fraction_bits + config.overflow_bits() - 1);
        while (fraction & top_bit) == 0 {
            fraction <<= 1;
            exponent -= 1;
        }
        fraction <<= 1;
        let mask = (1u64 << (config.fraction_bits + config.overflow_bits())) - 1;
        fraction &= mask;
    }
    let mut fraction_nibbles = config.fraction_nibbles() as usize;
    while fraction_nibbles > 0 && (fraction & 0xF) == 0 {
        fraction >>= 4;
        fraction_nibbles -= 1;
    }
    let mut text = String::new();
    if sign {
        text.push('-');
    }
    text.push_str("0x");
    text.push(if is_zero { '0' } else { '1' });
    if fraction_nibbles > 0 {
        text.push('.');
        let frac_str = format!("{:x}", fraction);
        if frac_str.len() < fraction_nibbles {
            let padding = fraction_nibbles - frac_str.len();
            for _ in 0..padding {
                text.push('0');
            }
        }
        text.push_str(&frac_str);
    }
    text.push('p');
    if exponent >= 0 {
        text.push('+');
    }
    text.push_str(&exponent.to_string());
    text
}

pub(super) fn format_operand(operand: &Operand, literal_format: LiteralFormat) -> String {
    match (literal_format, operand) {
        (LiteralFormat::Hexadecimal, Operand::LiteralBit32(value)) => {
            format!("0x{value:08x}")
        }
        (LiteralFormat::Hexadecimal, Operand::LiteralBit64(value)) => {
            format!("0x{value:016x}")
        }
        (LiteralFormat::Hexadecimal, Operand::LiteralExtInstInteger(value)) => {
            format!("0x{value:x}")
        }
        (_, Operand::ExecutionModel(model)) => {
            if let Some(name) = canonical_execution_model(*model) {
                name.to_string()
            } else {
                operand.disassemble()
            }
        }
        (_, Operand::LiteralString(value)) => format_literal_string(value),
        (_, Operand::StorageClass(class)) => {
            if let Some(name) = canonical_storage_class(*class) {
                name.to_string()
            } else {
                operand.disassemble()
            }
        }
        (_, Operand::MemoryAccess(_)) | (_, Operand::MemorySemantics(_)) => {
            normalize_mask_string(&operand.disassemble())
        }
        _ => operand.disassemble(),
    }
}

pub(super) fn normalize_mask_string(raw: &str) -> String {
    let mut seen = HashSet::new();
    let mut parts = Vec::new();
    for token in raw.split('|') {
        let trimmed = token.trim();
        let canonical = match trimmed {
            "MakePointerVisibleKHR" => "MakePointerVisible",
            "MakePointerAvailableKHR" => "MakePointerAvailable",
            "NonPrivatePointerKHR" => "NonPrivatePointer",
            "AcquireReleaseKHR" => "AcquireRelease",
            "AcquireKHR" => "Acquire",
            "ReleaseKHR" => "Release",
            other => other,
        };
        if seen.insert(canonical) {
            if canonical == trimmed {
                parts.push(trimmed.to_string());
            } else {
                parts.push(canonical.to_string());
            }
        }
    }
    parts.join("|")
}

pub(super) fn format_literal_string(value: &str) -> String {
    render_string_literal(value)
}

pub(super) fn canonical_execution_model(model: spirv::ExecutionModel) -> Option<&'static str> {
    use spirv::ExecutionModel;
    if model == ExecutionModel::RayGenerationNV || model == ExecutionModel::RayGenerationKHR {
        Some("RayGenerationKHR")
    } else if model == ExecutionModel::IntersectionNV || model == ExecutionModel::IntersectionKHR {
        Some("IntersectionKHR")
    } else if model == ExecutionModel::AnyHitNV || model == ExecutionModel::AnyHitKHR {
        Some("AnyHitKHR")
    } else if model == ExecutionModel::ClosestHitNV || model == ExecutionModel::ClosestHitKHR {
        Some("ClosestHitKHR")
    } else if model == ExecutionModel::MissNV || model == ExecutionModel::MissKHR {
        Some("MissKHR")
    } else if model == ExecutionModel::CallableNV || model == ExecutionModel::CallableKHR {
        Some("CallableKHR")
    } else {
        None
    }
}

pub(super) fn canonical_storage_class(class: spirv::StorageClass) -> Option<&'static str> {
    use spirv::StorageClass;
    if class == StorageClass::CallableDataNV || class == StorageClass::CallableDataKHR {
        Some("CallableDataKHR")
    } else if class == StorageClass::IncomingCallableDataNV
        || class == StorageClass::IncomingCallableDataKHR
    {
        Some("IncomingCallableDataKHR")
    } else if class == StorageClass::RayPayloadNV || class == StorageClass::RayPayloadKHR {
        Some("RayPayloadKHR")
    } else if class == StorageClass::HitAttributeNV || class == StorageClass::HitAttributeKHR {
        Some("HitAttributeKHR")
    } else if class == StorageClass::IncomingRayPayloadNV
        || class == StorageClass::IncomingRayPayloadKHR
    {
        Some("IncomingRayPayloadKHR")
    } else if class == StorageClass::ShaderRecordBufferNV
        || class == StorageClass::ShaderRecordBufferKHR
    {
        Some("ShaderRecordBufferKHR")
    } else if class == StorageClass::PhysicalStorageBuffer
        || class == StorageClass::PhysicalStorageBufferEXT
    {
        Some("PhysicalStorageBuffer")
    } else {
        None
    }
}
