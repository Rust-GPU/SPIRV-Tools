//! Parsing for image operations and image query operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{Op, Word};
use std::collections::HashMap;

use super::util::{parse_binary_args, parse_ternary_args, parse_unary_arg};

/// Unary image query operations.
const IMAGE_QUERY_UNARY: &[(&str, Op)] = &[
    ("ImageQuerySize", Op::ImageQuerySize),
    ("ImageQueryLevels", Op::ImageQueryLevels),
    ("ImageQuerySamples", Op::ImageQuerySamples),
];

/// Try to parse an image query operation.
pub fn try_parse_image(
    term: &str,
    result_id: Word,
    result_type: Word,
    id_map: &HashMap<String, Word>,
) -> Option<Instruction> {
    // Base ImageSample (no image operands) -> OpImageSampleImplicitLod
    if let Some(rest) = term.strip_prefix("(ImageSample ") {
        if let Some((image, coord)) = parse_binary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageSampleImplicitLod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                ],
            ));
        }
    }

    // ImageSampleOffset -> OpImageSampleImplicitLod with Offset
    if let Some(rest) = term.strip_prefix("(ImageSampleOffset ") {
        if let Some((image, coord, offset)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageSampleImplicitLod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                    rspirv::dr::Operand::ImageOperands(
                        rspirv::spirv::ImageOperands::OFFSET,
                    ),
                    rspirv::dr::Operand::IdRef(offset),
                ],
            ));
        }
    }

    // ImageSampleConstOffset -> OpImageSampleImplicitLod with ConstOffset
    if let Some(rest) = term.strip_prefix("(ImageSampleConstOffset ") {
        if let Some((image, coord, offset)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageSampleImplicitLod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                    rspirv::dr::Operand::ImageOperands(
                        rspirv::spirv::ImageOperands::CONST_OFFSET,
                    ),
                    rspirv::dr::Operand::IdRef(offset),
                ],
            ));
        }
    }

    // Base ImageFetch (no image operands) -> OpImageFetch
    if let Some(rest) = term.strip_prefix("(ImageFetch ") {
        if let Some((image, coord)) = parse_binary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageFetch,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                ],
            ));
        }
    }

    // ImageFetchOffset -> OpImageFetch with Offset
    if let Some(rest) = term.strip_prefix("(ImageFetchOffset ") {
        if let Some((image, coord, offset)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageFetch,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                    rspirv::dr::Operand::ImageOperands(
                        rspirv::spirv::ImageOperands::OFFSET,
                    ),
                    rspirv::dr::Operand::IdRef(offset),
                ],
            ));
        }
    }

    // ImageFetchConstOffset -> OpImageFetch with ConstOffset
    if let Some(rest) = term.strip_prefix("(ImageFetchConstOffset ") {
        if let Some((image, coord, offset)) = parse_ternary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageFetch,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                    rspirv::dr::Operand::ImageOperands(
                        rspirv::spirv::ImageOperands::CONST_OFFSET,
                    ),
                    rspirv::dr::Operand::IdRef(offset),
                ],
            ));
        }
    }

    // Unary image queries
    for (name, opcode) in IMAGE_QUERY_UNARY {
        let prefix = format!("({} ", name);
        if let Some(rest) = term.strip_prefix(&prefix) {
            if let Some(operand) = parse_unary_arg(rest, id_map) {
                return Some(Instruction::new(
                    *opcode,
                    Some(result_type),
                    Some(result_id),
                    vec![rspirv::dr::Operand::IdRef(operand)],
                ));
            }
        }
    }

    // ImageQuerySizeLod (image, lod) -> OpImageQuerySizeLod
    if let Some(rest) = term.strip_prefix("(ImageQuerySizeLod ") {
        if let Some((image, lod)) = parse_binary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageQuerySizeLod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(lod),
                ],
            ));
        }
    }

    // ImageQueryLod (sampled_image, coord) -> OpImageQueryLod
    if let Some(rest) = term.strip_prefix("(ImageQueryLod ") {
        if let Some((image, coord)) = parse_binary_args(rest, id_map) {
            return Some(Instruction::new(
                Op::ImageQueryLod,
                Some(result_type),
                Some(result_id),
                vec![
                    rspirv::dr::Operand::IdRef(image),
                    rspirv::dr::Operand::IdRef(coord),
                ],
            ));
        }
    }

    None
}
