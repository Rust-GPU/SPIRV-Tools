use core::convert::TryFrom;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::str::FromStr;

use rspirv::binary::Assemble;
use rspirv::dr::{self, Error as BuildError, InsertPoint};
use rspirv::grammar::{LogicalOperand, OperandKind, OperandQuantifier};
use rspirv::spirv;
use thiserror::Error;

use super::decoration::decoration_operand_descriptors;
use super::ext_inst::{lookup_custom_ext_inst_opcode, ExtInstImportInfo, ResolvedExtInst};
use super::instruction::{IdRef, LiteralNumber, ResultId, SpirvId, TypeId};
use super::options::TextToBinaryOptions;
use super::parser::{
    parse_instruction_with_origin, OperandValue, ParseError, ParsedInstruction, ParsedOperand,
};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;
use crate::string_literal::parse_string_literal;
use crate::target_env::TargetEnv;

use crate::validation::span::{SourceSpan, SpanMap};

/// Tracks textual identifiers and diagnostics while constructing a module.
#[derive(Debug)]
pub struct ModuleBuilder<'a> {
    named_ids: BTreeMap<&'a str, u32>,
    numeric_ids: BTreeMap<u32, u32>,
    next_numeric_id: u32,
    diagnostics: Vec<DiagnosticMessage<'static>>,
    value_types: BTreeMap<u32, u32>,
    composite_types: BTreeMap<u32, CompositeTypeInfo>,
    integer_constants: BTreeMap<u32, u64>,
    ext_inst_imports: BTreeMap<u32, ExtInstImportInfo>,
    preserve_numeric_ids: bool,
    /// Optional span map for tracking source locations of IDs.
    span_map: Option<SpanMap>,
}

#[derive(Debug, Error)]
enum MemberDecorationError {
    #[error("Decoration target must reference a type defined earlier")]
    UnknownType,
    #[error("Matrix layout decorations are only valid for struct members")]
    NotStruct,
    #[error("Struct member index {member_index} exceeds available field count {field_count}")]
    InvalidMemberIndex {
        member_index: usize,
        field_count: usize,
    },
}

/// Composite type metadata tracked by the assembler so diagnostics can reason about operand
/// layouts without falling back to the C++ implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeTypeInfo {
    /// Vector layout (component type + width).
    Vector(VectorTypeInfo),
    /// Array layout (element type + literal length).
    Array(ArrayTypeInfo),
    /// Struct layout (field list).
    Struct(StructTypeInfo),
    /// Matrix layout (column vector type + column count).
    Matrix(MatrixTypeInfo),
}

/// Describes a vector type tracked inside the module builder so we can validate operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorTypeInfo {
    component_type: u32,
    component_count: u32,
}

impl VectorTypeInfo {
    /// Creates a new vector descriptor capturing the component type and width.
    pub const fn new(component_type: u32, component_count: u32) -> Self {
        Self {
            component_type,
            component_count,
        }
    }

    /// Returns the component type id referenced by this vector.
    pub const fn component_type(self) -> u32 {
        self.component_type
    }

    /// Returns the number of components contained in the vector.
    pub const fn component_count(self) -> u32 {
        self.component_count
    }
}

/// Describes an array type (element type + length constant identifier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayTypeInfo {
    element_type: u32,
    length_constant: u32,
}

impl ArrayTypeInfo {
    /// Creates a new array descriptor capturing the element type and length constant id.
    pub const fn new(element_type: u32, length_constant: u32) -> Self {
        Self {
            element_type,
            length_constant,
        }
    }

    /// Returns the element type identifier encoded by this array.
    pub const fn element_type(self) -> u32 {
        self.element_type
    }

    /// Returns the identifier of the literal constant describing the array length.
    pub const fn length_constant(self) -> u32 {
        self.length_constant
    }
}

/// Describes a struct type using its field type list and member layout metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructTypeInfo {
    field_types: Vec<u32>,
    member_layouts: Vec<MemberLayout>,
}

/// Describes a matrix type tracked inside the module builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixTypeInfo {
    column_type: u32,
    column_count: u32,
}

impl StructTypeInfo {
    /// Creates a struct layout descriptor using the provided field type list.
    pub fn new(field_types: Vec<u32>) -> Self {
        let member_layouts = vec![MemberLayout::default(); field_types.len()];
        Self {
            field_types,
            member_layouts,
        }
    }

    /// Returns the field type at the given index if it exists.
    pub fn field_type(&self, index: usize) -> Option<u32> {
        self.field_types.get(index).copied()
    }

    /// Returns the number of fields contained in the struct.
    pub fn field_count(&self) -> usize {
        self.field_types.len()
    }

    /// Returns the layout metadata for a member if tracked.
    pub fn member_layout(&self, index: usize) -> Option<MemberLayout> {
        self.member_layouts.get(index).copied()
    }

    /// Returns mutable access to a member layout record.
    pub fn member_layout_mut(&mut self, index: usize) -> Option<&mut MemberLayout> {
        self.member_layouts.get_mut(index)
    }

    /// Returns all tracked member layouts.
    pub fn member_layouts(&self) -> &[MemberLayout] {
        &self.member_layouts
    }
}

impl MatrixTypeInfo {
    /// Creates a new matrix descriptor capturing the column vector type and count.
    pub const fn new(column_type: u32, column_count: u32) -> Self {
        Self {
            column_type,
            column_count,
        }
    }

    /// Returns the type id describing an individual column (which must be a vector).
    pub const fn column_type(self) -> u32 {
        self.column_type
    }

    /// Returns the number of columns contained within the matrix.
    pub const fn column_count(self) -> u32 {
        self.column_count
    }
}

/// Indicates whether a matrix is laid out row- or column-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixMajorness {
    RowMajor,
    ColumnMajor,
}

impl MatrixMajorness {
    fn as_str(self) -> &'static str {
        match self {
            MatrixMajorness::RowMajor => "RowMajor",
            MatrixMajorness::ColumnMajor => "ColMajor",
        }
    }
}

/// Captures matrix layout metadata attached to a struct member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemberLayout {
    majorness: Option<MemberMajorness>,
    matrix_stride: Option<MemberMatrixStride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemberMajorness {
    kind: MatrixMajorness,
    position: MessagePosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemberMatrixStride {
    value: u32,
    position: MessagePosition,
}

impl<'a> Default for ModuleBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ModuleBuilder<'a> {
    /// Creates a new builder that assigns numeric IDs starting at 1.
    pub fn new() -> Self {
        Self::with_numeric_preservation(false)
    }

    /// Creates a new builder and optionally preserves explicit numeric IDs.
    pub fn with_numeric_preservation(preserve_numeric_ids: bool) -> Self {
        Self::with_options(preserve_numeric_ids, false)
    }

    /// Creates a module builder with the given options.
    ///
    /// If `track_spans` is true, the builder will record source locations for all
    /// result IDs, which can be retrieved via `finish_with_spans`.
    pub fn with_options(preserve_numeric_ids: bool, track_spans: bool) -> Self {
        Self {
            named_ids: BTreeMap::new(),
            numeric_ids: BTreeMap::new(),
            next_numeric_id: 1,
            diagnostics: Vec::new(),
            value_types: BTreeMap::new(),
            composite_types: BTreeMap::new(),
            integer_constants: BTreeMap::new(),
            ext_inst_imports: BTreeMap::new(),
            preserve_numeric_ids,
            span_map: if track_spans {
                Some(SpanMap::new())
            } else {
                None
            },
        }
    }

    /// Returns true if explicit numeric identifiers should be preserved.
    pub fn preserve_numeric_ids(&self) -> bool {
        self.preserve_numeric_ids
    }

    /// Reserves an explicit numeric id so auto-assigned ids will not reuse it.
    pub fn reserve_numeric_id(&mut self, id: u32) {
        self.numeric_ids.entry(id).or_insert(id);
    }

    /// Resolves a result identifier to a numeric ID.
    ///
    /// If span tracking is enabled, this also records the source location
    /// where this result ID was defined.
    pub fn resolve_result_id(&mut self, id: ResultId<'a>) -> u32 {
        let numeric = self.resolve_spirv_id(id.as_spirv_id());
        // Record the span for this result ID
        self.record_id_span(numeric, SourceSpan::from_assembly_span(id.span()));
        numeric
    }

    /// Resolves a type identifier to a numeric ID.
    ///
    /// If span tracking is enabled, this also records the source location
    /// where this type ID was defined.
    pub fn resolve_type_id(&mut self, id: TypeId<'a>) -> u32 {
        let numeric = self.resolve_spirv_id(id.as_spirv_id());
        // Record the span for type definitions too
        self.record_id_span(numeric, SourceSpan::from_assembly_span(id.span()));
        numeric
    }

    /// Resolves an ID reference to a numeric ID.
    pub fn resolve_id_ref(&mut self, id: IdRef<'a>) -> u32 {
        self.resolve_spirv_id(id.as_spirv_id())
    }

    /// Records the instruction set kind for an `OpExtInstImport` result id.
    pub fn note_ext_inst_import(&mut self, id: u32, info: ExtInstImportInfo) {
        self.ext_inst_imports.insert(id, info);
    }

    /// Returns the recorded instruction set for an `OpExtInst` import if known.
    pub fn ext_inst_import(&self, id: u32) -> Option<&ExtInstImportInfo> {
        self.ext_inst_imports.get(&id)
    }

    /// Emits an assembler diagnostic.
    pub fn emit_error(&mut self, position: MessagePosition, message: impl Into<Cow<'static, str>>) {
        self.diagnostics.push(
            DiagnosticMessage::new(MessageLevel::Error, position, message).with_source("input"),
        );
    }

    /// Returns the collected diagnostics.
    pub fn diagnostics(&self) -> &[DiagnosticMessage<'static>] {
        &self.diagnostics
    }

    /// Consumes the builder and returns the diagnostics alongside the next ID bound.
    pub fn finish(mut self) -> (Vec<DiagnosticMessage<'static>>, u32) {
        self.validate_member_layouts();
        (self.diagnostics, self.next_numeric_id)
    }

    /// Consumes the builder and returns diagnostics, next ID bound, and optional span map.
    pub fn finish_with_spans(mut self) -> (Vec<DiagnosticMessage<'static>>, u32, Option<SpanMap>) {
        self.validate_member_layouts();
        (self.diagnostics, self.next_numeric_id, self.span_map)
    }

    /// Records the source span for a result ID if span tracking is enabled.
    pub fn record_id_span(&mut self, id: u32, span: SourceSpan) {
        if let Some(ref mut map) = self.span_map {
            map.record_id(id, span);
        }
    }

    fn resolve_spirv_id(&mut self, id: SpirvId<'a>) -> u32 {
        match id {
            SpirvId::Named(named) => {
                if let Some(existing) = self.named_ids.get(named.name()) {
                    *existing
                } else {
                    let allocated = self.allocate_fresh_id();
                    self.named_ids.insert(named.name(), allocated);
                    allocated
                }
            }
            SpirvId::Numeric(raw) => {
                let value = raw.get();
                if self.preserve_numeric_ids {
                    self.numeric_ids.entry(value).or_insert(value);
                    if self.next_numeric_id == value {
                        self.skip_reserved_ids();
                    }
                    value
                } else if let Some(existing) = self.numeric_ids.get(&value) {
                    *existing
                } else {
                    let allocated = self.allocate_fresh_id();
                    self.numeric_ids.insert(value, allocated);
                    allocated
                }
            }
        }
    }

    fn bind_result_id(&mut self, result_id: ResultId<'a>, numeric: u32) {
        self.bind_spirv_id(result_id.as_spirv_id(), numeric);
    }

    fn bind_spirv_id(&mut self, id: SpirvId<'a>, numeric: u32) {
        if let SpirvId::Named(named) = id {
            self.named_ids.insert(named.name(), numeric);
        } else if let SpirvId::Numeric(value) = id {
            if !self.preserve_numeric_ids {
                self.numeric_ids.insert(value.get(), numeric);
            }
        }
        self.next_numeric_id = self.next_numeric_id.max(numeric + 1);
    }

    fn allocate_fresh_id(&mut self) -> u32 {
        self.skip_reserved_ids();
        let allocated = self.next_numeric_id;
        self.next_numeric_id += 1;
        allocated
    }

    fn skip_reserved_ids(&mut self) {
        while self.numeric_ids.contains_key(&self.next_numeric_id) {
            self.next_numeric_id += 1;
        }
    }

    fn bind_typed_result(
        &mut self,
        result_type: TypeId<'a>,
        result_id: ResultId<'a>,
    ) -> (u32, u32) {
        let type_id = self.resolve_type_id(result_type);
        let value_id = self.resolve_result_id(result_id);
        self.value_types.insert(value_id, type_id);
        (type_id, value_id)
    }

    fn note_numeric_result_type(&mut self, value_id: u32, type_id: u32) {
        self.value_types.insert(value_id, type_id);
    }

    fn value_type(&self, value_id: u32) -> Option<u32> {
        self.value_types.get(&value_id).copied()
    }

    fn composite_type(&self, type_id: u32) -> Option<&CompositeTypeInfo> {
        self.composite_types.get(&type_id)
    }

    fn vector_type(&self, type_id: u32) -> Option<VectorTypeInfo> {
        self.composite_types
            .get(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Vector(vector) => Some(*vector),
                _ => None,
            })
    }

    fn note_vector_type(&mut self, type_id: u32, info: VectorTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Vector(info));
    }

    fn note_array_type(&mut self, type_id: u32, info: ArrayTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Array(info));
    }

    fn note_struct_type(&mut self, type_id: u32, info: StructTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Struct(info));
    }

    fn note_matrix_type(&mut self, type_id: u32, info: MatrixTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Matrix(info));
    }

    fn array_length(&self, info: &ArrayTypeInfo) -> Option<u32> {
        self.integer_constants
            .get(&info.length_constant())
            .and_then(|value| u32::try_from(*value).ok())
    }

    fn note_integer_constant(&mut self, result_id: u32, value: u64) {
        self.integer_constants.insert(result_id, value);
    }

    fn struct_info(&self, type_id: u32) -> Option<&StructTypeInfo> {
        self.composite_types
            .get(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Struct(struct_info) => Some(struct_info),
                _ => None,
            })
    }

    fn struct_info_mut(&mut self, type_id: u32) -> Option<&mut StructTypeInfo> {
        self.composite_types
            .get_mut(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Struct(struct_info) => Some(struct_info),
                _ => None,
            })
    }

    fn resolve_struct_member_type(
        &self,
        type_id: u32,
        member_index: usize,
    ) -> Result<u32, MemberDecorationError> {
        match self.composite_types.get(&type_id) {
            Some(CompositeTypeInfo::Struct(info)) => {
                info.field_type(member_index)
                    .ok_or(MemberDecorationError::InvalidMemberIndex {
                        member_index,
                        field_count: info.field_count(),
                    })
            }
            Some(_) => Err(MemberDecorationError::NotStruct),
            None => Err(MemberDecorationError::UnknownType),
        }
    }

    fn type_contains_matrix(&self, type_id: u32) -> bool {
        match self.composite_types.get(&type_id) {
            Some(CompositeTypeInfo::Matrix(_)) => true,
            Some(CompositeTypeInfo::Array(info)) => self.type_contains_matrix(info.element_type()),
            _ => false,
        }
    }

    fn apply_member_majorness(
        &mut self,
        type_id: u32,
        member_index: usize,
        majorness: MatrixMajorness,
        position: MessagePosition,
    ) {
        let member_type = match self.resolve_struct_member_type(type_id, member_index) {
            Ok(found) => found,
            Err(error) => {
                self.emit_member_decoration_error(error, position);
                return;
            }
        };
        if !self.type_contains_matrix(member_type) {
            self.emit_error(
                position,
                format!(
                    "{} decoration requires the member type to be a matrix or array of matrices",
                    majorness.as_str()
                ),
            );
            return;
        }

        let mut error: Option<String> = None;
        if let Some(info) = self.struct_info_mut(type_id) {
            if let Some(layout) = info.member_layout_mut(member_index) {
                if let Some(existing) = layout.majorness {
                    if existing.kind == majorness {
                        error = Some(format!(
                            "{} decoration already specified for this member",
                            majorness.as_str()
                        ));
                    } else {
                        error = Some(
                            "RowMajor and ColMajor decorations cannot both target the same member"
                                .to_string(),
                        );
                    }
                } else {
                    layout.majorness = Some(MemberMajorness {
                        kind: majorness,
                        position,
                    });
                }
            }
        }

        if let Some(message) = error {
            self.emit_error(position, message);
        }
    }

    fn apply_member_matrix_stride(
        &mut self,
        type_id: u32,
        member_index: usize,
        stride: u32,
        position: MessagePosition,
    ) {
        let member_type = match self.resolve_struct_member_type(type_id, member_index) {
            Ok(found) => found,
            Err(error) => {
                self.emit_member_decoration_error(error, position);
                return;
            }
        };

        if !self.type_contains_matrix(member_type) {
            self.emit_error(
                position,
                "MatrixStride decoration requires the member type to contain a matrix",
            );
            return;
        }

        if stride == 0 {
            self.emit_error(position, "MatrixStride must be greater than zero");
            return;
        }

        let mut error: Option<String> = None;
        if let Some(info) = self.struct_info_mut(type_id) {
            if let Some(layout) = info.member_layout_mut(member_index) {
                if layout.matrix_stride.is_some() {
                    error =
                        Some("MatrixStride decoration already specified for this member".into());
                } else {
                    layout.matrix_stride = Some(MemberMatrixStride {
                        value: stride,
                        position,
                    });
                }
            }
        }

        if let Some(message) = error {
            self.emit_error(position, message);
        }
    }

    fn emit_member_decoration_error(
        &mut self,
        error: MemberDecorationError,
        position: MessagePosition,
    ) {
        self.emit_error(position, error.to_string());
    }

    fn validate_member_layouts(&mut self) {
        let struct_ids: Vec<u32> = self
            .composite_types
            .iter()
            .filter_map(|(id, info)| matches!(info, CompositeTypeInfo::Struct(_)).then_some(*id))
            .collect();
        for struct_id in struct_ids {
            let layouts = match self.struct_info(struct_id) {
                Some(info) => info.member_layouts().to_vec(),
                None => continue,
            };
            for layout in layouts {
                if let Some(majorness) = layout.majorness {
                    if layout.matrix_stride.is_none() {
                        self.emit_error(
                            majorness.position,
                            format!(
                                "{} decoration requires an accompanying MatrixStride",
                                majorness.kind.as_str()
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// Error emitted when the assembler produces diagnostics instead of a finished module.
#[derive(Debug, Error)]
#[error("assembly failed with diagnostics")]
pub struct AssemblyError {
    diagnostics: Vec<DiagnosticMessage<'static>>,
}

impl AssemblyError {
    fn new(diagnostics: Vec<DiagnosticMessage<'static>>) -> Self {
        Self { diagnostics }
    }

    /// Borrows the underlying diagnostics describing the failure.
    pub fn diagnostics(&self) -> &[DiagnosticMessage<'static>] {
        &self.diagnostics
    }

    /// Consumes this error and returns the owned diagnostics.
    pub fn into_diagnostics(self) -> Vec<DiagnosticMessage<'static>> {
        self.diagnostics
    }
}

/// Drives translation from parsed instructions to SPIR-V DR form.
pub struct AssemblyTranslator<'a> {
    module_builder: ModuleBuilder<'a>,
    builder: dr::Builder,
}

impl<'a> Default for AssemblyTranslator<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> AssemblyTranslator<'a> {
    /// Creates a new translator with an empty module.
    pub fn new() -> Self {
        Self::with_target_env(TargetEnv::Universal1_6)
    }

    /// Creates a new translator configured for the requested target environment.
    pub fn with_target_env(env: TargetEnv) -> Self {
        Self::with_target_env_and_options(env, TextToBinaryOptions::NONE)
    }

    /// Creates a translator configured for the provided environment and options.
    pub fn with_target_env_and_options(env: TargetEnv, options: TextToBinaryOptions) -> Self {
        Self::with_full_options(env, options, false)
    }

    /// Creates a translator with full control over all options including span tracking.
    pub fn with_full_options(
        env: TargetEnv,
        options: TextToBinaryOptions,
        track_spans: bool,
    ) -> Self {
        let mut builder = dr::Builder::new();
        configure_builder_for_env(&mut builder, env);
        let preserve_numeric_ids = options.contains(TextToBinaryOptions::PRESERVE_NUMERIC_IDS);
        Self {
            module_builder: ModuleBuilder::with_options(preserve_numeric_ids, track_spans),
            builder,
        }
    }

    fn reserve_numeric_result_ids(&mut self, instructions: &[ParsedInstruction<'a>]) {
        if !self.module_builder.preserve_numeric_ids() {
            return;
        }
        for instruction in instructions {
            if let Some(result) = instruction.result_id() {
                if let Some(numeric) = result.as_spirv_id().as_numeric() {
                    self.module_builder.reserve_numeric_id(numeric.get());
                }
            }
        }
    }

    /// Translates a parsed instruction.
    pub fn translate(&mut self, instruction: &ParsedInstruction<'a>) {
        match instruction.opcode() {
            spirv::Op::Capability => self.translate_capability(instruction),
            spirv::Op::Extension => self.translate_extension(instruction),
            spirv::Op::ConditionalExtensionINTEL => {
                self.translate_conditional_extension(instruction)
            }
            spirv::Op::ExtInstImport => self.translate_ext_inst_import(instruction),
            spirv::Op::TypeVoid => self.translate_type_void(instruction),
            spirv::Op::TypeBool => self.translate_type_bool(instruction),
            spirv::Op::TypeInt => self.translate_type_int(instruction),
            spirv::Op::TypeFloat => self.translate_type_float(instruction),
            spirv::Op::TypeFunction => self.translate_type_function(instruction),
            spirv::Op::TypePointer => self.translate_type_pointer(instruction),
            spirv::Op::TypeVector => self.translate_type_vector(instruction),
            spirv::Op::TypeArray => self.translate_type_array(instruction),
            spirv::Op::TypeStruct => self.translate_type_struct(instruction),
            spirv::Op::TypeMatrix => self.translate_type_matrix(instruction),
            spirv::Op::MemoryModel => self.translate_memory_model(instruction),
            spirv::Op::EntryPoint => self.translate_entry_point(instruction),
            spirv::Op::ExecutionMode => self.translate_execution_mode(instruction),
            spirv::Op::Name => self.translate_name(instruction),
            spirv::Op::MemberName => self.translate_member_name(instruction),
            spirv::Op::AccessChain => self.translate_access_chain(instruction, false),
            spirv::Op::InBoundsAccessChain => self.translate_access_chain(instruction, true),
            spirv::Op::CopyMemory => self.translate_copy_memory(instruction, false),
            spirv::Op::CopyMemorySized => self.translate_copy_memory(instruction, true),
            spirv::Op::SelectionMerge => self.translate_selection_merge(instruction),
            spirv::Op::LoopMerge => self.translate_loop_merge(instruction),
            spirv::Op::Decorate => self.translate_decorate(instruction),
            spirv::Op::DecorateId => self.translate_decorate_id(instruction),
            spirv::Op::DecorateString => self.translate_decorate_string(instruction),
            spirv::Op::MemberDecorate => self.translate_member_decorate(instruction),
            spirv::Op::MemberDecorateString => self.translate_member_decorate_string(instruction),
            spirv::Op::CompositeConstruct => self.translate_composite_construct(instruction),
            spirv::Op::VectorShuffle => self.translate_vector_shuffle(instruction),
            spirv::Op::CompositeExtract => self.translate_composite_extract(instruction),
            spirv::Op::CompositeInsert => self.translate_composite_insert(instruction),
            spirv::Op::ExtInst => self.translate_ext_inst(instruction),
            spirv::Op::Phi => self.translate_phi(instruction),
            spirv::Op::ConstantTrue => self.translate_boolean_constant(instruction, true),
            spirv::Op::ConstantFalse => self.translate_boolean_constant(instruction, false),
            spirv::Op::ConstantComposite => self.translate_constant_composite(instruction),
            spirv::Op::ConstantNull => self.translate_constant_null(instruction),
            spirv::Op::Function => self.translate_function(instruction),
            spirv::Op::FunctionParameter => self.translate_function_parameter(instruction),
            spirv::Op::Label => self.translate_label(instruction),
            spirv::Op::Branch => self.translate_branch(instruction),
            spirv::Op::BranchConditional => self.translate_branch_conditional(instruction),
            spirv::Op::Return => self.translate_return(instruction),
            spirv::Op::ReturnValue => self.translate_return_value(instruction),
            spirv::Op::FunctionEnd => self.translate_function_end(instruction),
            spirv::Op::Constant => self.translate_constant(instruction),
            spirv::Op::Variable => self.translate_variable(instruction),
            spirv::Op::Load => self.translate_load(instruction),
            spirv::Op::Store => self.translate_store(instruction),
            spirv::Op::IAdd
            | spirv::Op::ISub
            | spirv::Op::IMul
            | spirv::Op::FAdd
            | spirv::Op::FSub
            | spirv::Op::FMul => self.translate_binary_arithmetic(instruction),
            spirv::Op::Bitcast
            | spirv::Op::ConvertFToU
            | spirv::Op::ConvertFToS
            | spirv::Op::ConvertSToF
            | spirv::Op::ConvertUToF
            | spirv::Op::UConvert
            | spirv::Op::SConvert
            | spirv::Op::FConvert => self.translate_unary_op(instruction),
            _ => self
                .module_builder
                .emit_error(instruction.opcode_position(), "unsupported opcode"),
        }
    }

    fn record_instruction(&mut self, _inst: dr::Instruction) {}

    fn record_from_module<F>(&mut self, fetch: F)
    where
        F: FnOnce(&dr::Module) -> Option<dr::Instruction>,
    {
        if let Some(inst) = fetch(self.builder.module_ref()) {
            self.record_instruction(inst);
        }
    }

    fn record_from_current_block(&mut self) {
        if let (Some(function), Some(block)) = (
            self.builder.selected_function(),
            self.builder.selected_block(),
        ) {
            self.record_from_module(|module| {
                module
                    .functions
                    .get(function)
                    .and_then(|f| f.blocks.get(block))
                    .and_then(|bb| bb.instructions.last())
                    .cloned()
            });
        }
    }

    fn record_function_def(&mut self) {}

    fn record_function_param(&mut self) {
        if let Some(function) = self.builder.selected_function() {
            self.record_from_module(|module| {
                module
                    .functions
                    .get(function)
                    .and_then(|f| f.parameters.last())
                    .cloned()
            });
        }
    }

    fn translate_capability(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCapability missing enumerant");
            return;
        };
        let Some(capability) =
            self.parse_enum_operand::<spirv::Capability>(Some(operand), "capability", opcode_pos)
        else {
            return;
        };
        self.builder.capability(capability);
        self.record_from_module(|module| module.capabilities.last().cloned());
    }

    fn translate_extension(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        if instruction.result_id().is_some() {
            self.module_builder
                .emit_error(opcode_pos, "OpExtension does not produce a result id");
            return;
        }
        let Some(name_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtension missing extension name");
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpExtension operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = instruction.operands().get(1) {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpExtension received unexpected operands",
            );
            return;
        }
        self.builder.extension(name);
        self.record_from_module(|module| module.extensions.last().cloned());
    }

    fn translate_conditional_extension(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        if instruction.result_id().is_some() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL does not produce a result id",
            );
            return;
        }
        let mut operands = instruction.operands().iter();
        let Some(condition_operand) = operands.next() else {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL missing condition id",
            );
            return;
        };
        let Some(extension_operand) = operands.next() else {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConditionalExtensionINTEL missing extension name",
            );
            return;
        };
        let Some(condition_id) =
            self.operand_as_id(condition_operand, "OpConditionalExtensionINTEL condition")
        else {
            return;
        };
        let extension = match extension_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    extension_operand.span().start(),
                    "OpConditionalExtensionINTEL extension must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpConditionalExtensionINTEL received unexpected operands",
            );
            return;
        }
        let inst = dr::Instruction::new(
            spirv::Op::ConditionalExtensionINTEL,
            None,
            None,
            vec![
                dr::Operand::IdRef(condition_id),
                dr::Operand::LiteralString(extension),
            ],
        );
        self.builder.module_mut().extensions.push(inst);
        self.record_from_module(|module| module.extensions.last().cloned());
    }

    fn translate_ext_inst_import(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInstImport missing result id");
            return;
        };
        let Some(name_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInstImport missing import name");
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpExtInstImport operand must be a literal string",
                );
                return;
            }
        };
        if instruction.operands().len() > 1 {
            if let Some(extra) = instruction.operands().get(1) {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "OpExtInstImport only accepts a single operand",
                );
            }
            return;
        }
        let numeric_id = self.module_builder.resolve_result_id(result_id);
        let info = ExtInstImportInfo::new(&name);
        let inst = dr::Instruction::new(
            spirv::Op::ExtInstImport,
            None,
            Some(numeric_id),
            vec![dr::Operand::LiteralString(name)],
        );
        self.builder.module_mut().ext_inst_imports.push(inst);
        self.module_builder.note_ext_inst_import(numeric_id, info);
        self.record_from_module(|module| module.ext_inst_imports.last().cloned());
    }

    fn translate_type_void(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVoid requires a result id");
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_void_id(Some(result_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_bool(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeBool requires a result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeBool does not take operands");
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_bool_id(Some(result_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_int(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt requires a result identifier");
            return;
        };

        let mut operands = instruction.operands().iter();
        let Some(width_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt missing width literal");
            return;
        };
        let width_literal = match width_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    width_operand.span().start(),
                    "Width operand must be literal",
                );
                return;
            }
        };
        let Some(signed_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeInt missing signedness literal");
            return;
        };
        let signed_literal = match signed_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    signed_operand.span().start(),
                    "Signedness operand must be literal",
                );
                return;
            }
        };

        let width = literal_to_u32(width_literal);
        let signedness = literal_to_u32(signed_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_int_id(Some(result_id), width, signedness);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_float(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFloat requires a result identifier");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(width_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFloat missing width literal");
            return;
        };
        let width_literal = match width_operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    width_operand.span().start(),
                    "Width operand must be literal",
                );
                return;
            }
        };
        let encoding = match operands.next() {
            Some(operand) => match self.parse_fp_encoding_operand(operand) {
                Some(value) => Some(value),
                None => return,
            },
            None => None,
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeFloat received unexpected operands",
            );
            return;
        }
        let width = literal_to_u32(width_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder.type_float_id(Some(result_id), width, encoding);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstant missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstant requires a result identifier");
            return;
        };

        let literal_operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpConstant missing literal operand");
                return;
            }
        };

        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);

        // Look up the result type to determine how to encode the literal.
        // The operand is a LiteralContextDependentNumber: its encoding depends
        // on the result type, matching the C++ spirv-as behavior.
        let float_width = self.lookup_float_type_width(type_id);

        let operands = match literal_operand.value() {
            OperandValue::Literal(literal) => {
                // Integer text (e.g. "42", "-1"). For float types, convert the
                // integer value to float then encode as IEEE 754 bit pattern.
                match float_width {
                    Some(32) => {
                        let float_val = match literal {
                            LiteralNumber::Unsigned(v) => *v as f32,
                            LiteralNumber::Signed(v) => *v as f32,
                        };
                        vec![dr::Operand::LiteralBit32(float_val.to_bits())]
                    }
                    Some(64) => {
                        let float_val = match literal {
                            LiteralNumber::Unsigned(v) => *v as f64,
                            LiteralNumber::Signed(v) => *v as f64,
                        };
                        vec![dr::Operand::LiteralBit64(float_val.to_bits())]
                    }
                    Some(_) => {
                        // 16-bit or other widths: encode as raw bits in a 32-bit word
                        vec![dr::Operand::LiteralBit32(literal_to_u32(literal))]
                    }
                    None => {
                        // Integer type: encode as raw integer bits.
                        self.module_builder
                            .note_integer_constant(result_id, literal_to_u64(literal));
                        vec![encode_literal_operand(literal)]
                    }
                }
            }
            OperandValue::Word(word) => {
                // Non-integer text (e.g. "42.5", "0x1.8p+1"). Must be a float type.
                let text = word.as_str();
                if float_width.is_none() {
                    self.module_builder.emit_error(
                        literal_operand.span().start(),
                        "integer type requires an integer literal",
                    );
                    return;
                }
                match float_width {
                    Some(32) => match text.parse::<f32>() {
                        Ok(val) => vec![dr::Operand::LiteralBit32(val.to_bits())],
                        Err(_) => {
                            self.module_builder.emit_error(
                                literal_operand.span().start(),
                                "invalid 32-bit float literal",
                            );
                            return;
                        }
                    },
                    Some(64) => match text.parse::<f64>() {
                        Ok(val) => vec![dr::Operand::LiteralBit64(val.to_bits())],
                        Err(_) => {
                            self.module_builder.emit_error(
                                literal_operand.span().start(),
                                "invalid 64-bit float literal",
                            );
                            return;
                        }
                    },
                    _ => {
                        self.module_builder.emit_error(
                            literal_operand.span().start(),
                            "unsupported float width for literal",
                        );
                        return;
                    }
                }
            }
            _ => {
                self.module_builder.emit_error(
                    literal_operand.span().start(),
                    "OpConstant literal operand must be numeric",
                );
                return;
            }
        };

        let inst = dr::Instruction::new(
            spirv::Op::Constant,
            Some(type_id),
            Some(result_id),
            operands,
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    /// Look up whether `type_id` refers to an `OpTypeFloat` instruction
    /// and return its bit width if so.
    fn lookup_float_type_width(&self, type_id: u32) -> Option<u32> {
        self.builder
            .module_ref()
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == spirv::Op::TypeFloat && inst.result_id == Some(type_id)
            })
            .and_then(|inst| inst.operands.first())
            .and_then(|op| match op {
                dr::Operand::LiteralBit32(w) => Some(*w),
                _ => None,
            })
    }

    fn translate_type_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeFunction missing result id");
            return;
        };

        let mut operands = instruction.operands().iter();
        let return_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpTypeFunction missing return type");
                return;
            }
        };
        let return_type = match return_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    return_operand.span().start(),
                    "Return type must be an id reference",
                );
                return;
            }
        };

        let mut parameter_types = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => {
                    parameter_types.push(self.module_builder.resolve_id_ref(*id))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Parameter type must be an id reference",
                    );
                    return;
                }
            }
        }

        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_function_id(Some(result_id), return_type, parameter_types);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_function(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunction missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunction missing result id");
            return;
        };

        let mut operands = instruction.operands().iter();
        let control_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpFunction missing control operand");
                return;
            }
        };
        let control = match self.parse_function_control(control_operand) {
            Some(value) => value,
            None => return,
        };
        let function_type_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpFunction missing function type");
                return;
            }
        };
        let function_type = match function_type_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    function_type_operand.span().start(),
                    "Function type must be an id reference",
                );
                return;
            }
        };

        let (result_type, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self
            .builder
            .begin_function(result_type, Some(result_id), control, function_type)
        {
            Ok(_) => self.record_function_def(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_function_parameter(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunctionParameter missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpFunctionParameter missing result id");
            return;
        };

        let type_id = self.module_builder.resolve_type_id(result_type);
        match self.builder.function_parameter(type_id) {
            Ok(parameter_id) => {
                self.module_builder.bind_result_id(result_id, parameter_id);
                self.module_builder
                    .note_numeric_result_type(parameter_id, type_id);
                self.record_function_param();
            }
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_label(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLabel missing result id");
            return;
        };
        let label_id = self.module_builder.resolve_result_id(result_id);
        match self.builder.begin_block(Some(label_id)) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_return(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        match self.builder.ret() {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_return_value(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let operand = match instruction.operands().first() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpReturnValue missing result id");
                return;
            }
        };
        let Some(value_id) = self.operand_as_id(operand, "return value") else {
            return;
        };
        match self.builder.ret_value(value_id) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, operand.span().start()),
        }
    }

    fn translate_function_end(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let current_function = self.builder.selected_function();
        match self.builder.end_function() {
            Ok(_) => {
                if let Some(function) = current_function {
                    self.record_from_module(|module| {
                        module
                            .functions
                            .get(function)
                            .and_then(|f| f.end.as_ref())
                            .cloned()
                    });
                }
            }
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_type_pointer(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing storage class");
            return;
        };
        let Some(storage_class) = self.parse_enum_operand::<spirv::StorageClass>(
            Some(storage_operand),
            "storage class",
            opcode_pos,
        ) else {
            return;
        };
        let Some(pointee_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypePointer missing pointee type");
            return;
        };
        let Some(pointee_id) = self.operand_as_id(pointee_operand, "pointee type") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypePointer received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_pointer(Some(result_id), storage_class, pointee_id);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_vector(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(component_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing component type");
            return;
        };
        let Some(component_type) = self.operand_as_id(component_operand, "component type") else {
            return;
        };
        let Some(count_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeVector missing component count");
            return;
        };
        let count_literal = match count_operand.value() {
            OperandValue::Literal(literal) => literal,
            _ => {
                self.module_builder.emit_error(
                    count_operand.span().start(),
                    "Component count must be a literal",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeVector received unexpected operands",
            );
            return;
        }
        let component_count = literal_to_u32(count_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        self.builder
            .type_vector_id(Some(result_id), component_type, component_count);
        self.module_builder.note_vector_type(
            result_id,
            VectorTypeInfo::new(component_type, component_count),
        );
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_array(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(element_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing element type");
            return;
        };
        let Some(element_type) = self.operand_as_id(element_operand, "element type") else {
            return;
        };
        let Some(length_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeArray missing length id");
            return;
        };
        let Some(length_id) = self.operand_as_id(length_operand, "length id") else {
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeArray received unexpected operands",
            );
            return;
        }
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeArray,
            None,
            Some(result_id),
            vec![
                dr::Operand::IdRef(element_type),
                dr::Operand::IdRef(length_id),
            ],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_array_type(result_id, ArrayTypeInfo::new(element_type, length_id));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_struct(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeStruct missing result id");
            return;
        };
        let result_id = self.module_builder.resolve_result_id(result_id);
        let mut field_types = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => field_types.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Struct members must be id references",
                    );
                    return;
                }
            }
        }
        let operands = field_types
            .iter()
            .copied()
            .map(dr::Operand::IdRef)
            .collect();
        let inst = dr::Instruction::new(spirv::Op::TypeStruct, None, Some(result_id), operands);
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_struct_type(result_id, StructTypeInfo::new(field_types));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_type_matrix(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(column_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing column type");
            return;
        };
        let Some(column_type) = self.operand_as_id(column_operand, "column type") else {
            return;
        };
        if self.module_builder.vector_type(column_type).is_none() {
            self.module_builder.emit_error(
                column_operand.span().start(),
                "Matrix column type must be a previously defined vector",
            );
            return;
        }
        let Some(count_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpTypeMatrix missing column count");
            return;
        };
        let column_count_literal = match count_operand.value() {
            OperandValue::Literal(literal) => literal,
            _ => {
                self.module_builder.emit_error(
                    count_operand.span().start(),
                    "Matrix column count must be a literal",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpTypeMatrix received unexpected operands",
            );
            return;
        }
        let column_count = literal_to_u32(column_count_literal);
        let result_id = self.module_builder.resolve_result_id(result_id);
        let inst = dr::Instruction::new(
            spirv::Op::TypeMatrix,
            None,
            Some(result_id),
            vec![
                dr::Operand::IdRef(column_type),
                dr::Operand::LiteralBit32(column_count),
            ],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.module_builder
            .note_matrix_type(result_id, MatrixTypeInfo::new(column_type, column_count));
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_memory_model(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let addressing = match self.parse_enum_operand::<spirv::AddressingModel>(
            operands.next(),
            "addressing model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };
        let memory = match self.parse_enum_operand::<spirv::MemoryModel>(
            operands.next(),
            "memory model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };
        self.builder.memory_model(addressing, memory);
        self.record_from_module(|module| module.memory_model.clone());
    }

    fn translate_entry_point(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let execution_model = match self.parse_enum_operand::<spirv::ExecutionModel>(
            operands.next(),
            "execution model",
            opcode_pos,
        ) {
            Some(value) => value,
            None => return,
        };

        let function_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpEntryPoint missing function identifier");
                return;
            }
        };
        let function_id = match function_operand.value() {
            OperandValue::Id(id) => self.module_builder.resolve_id_ref(*id),
            _ => {
                self.module_builder.emit_error(
                    function_operand.span().start(),
                    "Function operand must be an id reference",
                );
                return;
            }
        };

        let name_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpEntryPoint missing name literal");
                return;
            }
        };
        let entry_name = match name_operand.value() {
            OperandValue::String(name) => parse_string_literal(name),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "Entry point name must be a literal string",
                );
                return;
            }
        };

        let mut interfaces = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => interfaces.push(self.module_builder.resolve_id_ref(*id)),
                _ => self.module_builder.emit_error(
                    operand.span().start(),
                    "Interface operand must be an id reference",
                ),
            }
        }

        self.builder
            .entry_point(execution_model, function_id, &entry_name, interfaces);
        self.record_from_module(|module| module.entry_points.last().cloned());
    }

    fn translate_name(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpName missing target id");
            return;
        };
        let Some(name_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpName missing name operand");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target id") else {
            return;
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpName operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder
                .emit_error(extra.span().start(), "OpName received unexpected operands");
            return;
        }
        self.builder.name(target_id, name);
        self.record_from_module(|module| module.debug_names.last().cloned());
    }

    fn translate_member_name(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing target id");
            return;
        };
        let Some(member_index_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing member index operand");
            return;
        };
        let Some(name_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberName missing name operand");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target id") else {
            return;
        };
        let member_index = match member_index_operand.value() {
            OperandValue::Literal(literal) => literal_to_u32(literal),
            _ => {
                self.module_builder.emit_error(
                    member_index_operand.span().start(),
                    "OpMemberName member index must be a literal number",
                );
                return;
            }
        };
        let name = match name_operand.value() {
            OperandValue::String(value) => parse_string_literal(value),
            _ => {
                self.module_builder.emit_error(
                    name_operand.span().start(),
                    "OpMemberName operand must be a literal string",
                );
                return;
            }
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpMemberName received unexpected operands",
            );
            return;
        }
        self.builder.member_name(target_id, member_index, name);
        self.record_from_module(|module| module.debug_names.last().cloned());
    }

    fn translate_access_chain(&mut self, instruction: &ParsedInstruction<'a>, in_bounds: bool) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(base_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Access chain missing base pointer");
            return;
        };
        let Some(base_id) = self.operand_as_id(base_operand, "base pointer") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Id(id) => {
                    indexes.push(self.module_builder.resolve_id_ref(*id));
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Access chain indexes must be ids");
                    return;
                }
            }
        }
        let (result_type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let result = if in_bounds {
            self.builder
                .in_bounds_access_chain(result_type_id, Some(result_id), base_id, indexes)
        } else {
            self.builder
                .access_chain(result_type_id, Some(result_id), base_id, indexes)
        };
        match result {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_execution_mode(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let entry_operand = match operands.next() {
            Some(operand) => operand,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpExecutionMode missing entry point");
                return;
            }
        };
        let Some(entry_point) = self.operand_as_id(entry_operand, "entry point") else {
            return;
        };
        let Some(execution_mode) = self.parse_enum_operand::<spirv::ExecutionMode>(
            operands.next(),
            "execution mode",
            opcode_pos,
        ) else {
            return;
        };
        let mut parameters = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => parameters.push(literal_to_u32(literal)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Execution mode parameters must be literals",
                    );
                    return;
                }
            }
        }
        self.builder
            .execution_mode(entry_point, execution_mode, parameters);
        self.record_from_module(|module| module.execution_modes.last().cloned());
    }

    fn translate_selection_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpSelectionMerge missing merge block");
            return;
        };
        let Some(merge_id) = self.operand_as_id(merge_operand, "merge block") else {
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpSelectionMerge missing control mask");
            return;
        };
        let Some(selection_control) = self.parse_selection_control(control_operand) else {
            return;
        };
        match self.builder.selection_merge(merge_id, selection_control) {
            Ok(()) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, merge_operand.span().start()),
        }
    }

    fn translate_loop_merge(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(merge_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing merge block");
            return;
        };
        let Some(continue_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing continue target");
            return;
        };
        let Some(control_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoopMerge missing control mask");
            return;
        };
        let Some(merge_id) = self.operand_as_id(merge_operand, "merge block") else {
            return;
        };
        let Some(continue_id) = self.operand_as_id(continue_operand, "continue target") else {
            return;
        };
        let Some((loop_control, control_operands)) =
            self.parse_loop_control_operand(control_operand, &mut operands)
        else {
            return;
        };
        match self
            .builder
            .loop_merge(merge_id, continue_id, loop_control, control_operands)
        {
            Ok(()) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, merge_operand.span().start()),
        }
    }

    fn translate_composite_construct(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeConstruct missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeConstruct missing result id");
            return;
        };
        if instruction.operands().is_empty() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpCompositeConstruct requires at least one constituent",
            );
            return;
        }
        let mut constituents = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => constituents.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Constituent must be an id reference",
                    );
                    return;
                }
            }
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self
            .builder
            .composite_construct(type_id, Some(result_id), constituents)
        {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_vector_shuffle(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(vector1_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing first vector");
            return;
        };
        let Some(vector2_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle missing second vector");
            return;
        };
        let Some(vector1_id) = self.operand_as_id(vector1_operand, "first vector") else {
            return;
        };
        let Some(vector2_id) = self.operand_as_id(vector2_operand, "second vector") else {
            return;
        };
        let mut components = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    components.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Shuffle components must be literals",
                    );
                    return;
                }
            }
        }
        let component_position = components
            .first()
            .map(|(_, span)| *span)
            .unwrap_or_else(|| vector2_operand.span().start());
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(result_vector) = self.module_builder.vector_type(type_id) else {
            self.module_builder
                .emit_error(opcode_pos, "OpVectorShuffle result type must be a vector");
            return;
        };
        let expected_components = result_vector.component_count() as usize;
        if components.len() != expected_components {
            self.module_builder.emit_error(
                component_position,
                format!(
                    "OpVectorShuffle expects {expected_components} component literals but received {}",
                    components.len()
                ),
            );
            return;
        }

        let Some(vector1_type_id) = self.module_builder.value_type(vector1_id) else {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand has no known type",
            );
            return;
        };
        let Some(vector1_info) = self.module_builder.vector_type(vector1_type_id) else {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand must be a vector",
            );
            return;
        };
        let Some(vector2_type_id) = self.module_builder.value_type(vector2_id) else {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand has no known type",
            );
            return;
        };
        let Some(vector2_info) = self.module_builder.vector_type(vector2_type_id) else {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand must be a vector",
            );
            return;
        };
        if vector1_info.component_type() != result_vector.component_type() {
            self.module_builder.emit_error(
                vector1_operand.span().start(),
                "First vector operand type does not match the result vector component type",
            );
            return;
        }
        if vector2_info.component_type() != result_vector.component_type() {
            self.module_builder.emit_error(
                vector2_operand.span().start(),
                "Second vector operand type does not match the result vector component type",
            );
            return;
        }

        let total_components =
            u64::from(vector1_info.component_count()) + u64::from(vector2_info.component_count());
        for (value, position) in &components {
            if *value != u32::MAX && u64::from(*value) >= total_components {
                self.module_builder.emit_error(
                    *position,
                    format!(
                        "Shuffle component {value} exceeds the available inputs ({total_components})",
                    ),
                );
                return;
            }
        }

        let literal_components: Vec<u32> = components.iter().map(|(value, _)| *value).collect();
        if let Err(error) = self.builder.vector_shuffle(
            type_id,
            Some(result_id),
            vector1_id,
            vector2_id,
            literal_components,
        ) {
            self.emit_builder_error(error, opcode_pos);
        } else {
            self.record_from_current_block();
        }
    }

    fn translate_boolean_constant(&mut self, instruction: &ParsedInstruction<'a>, value: bool) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constant missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constant missing result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "Boolean constants do not take operands");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let opcode = if value {
            spirv::Op::ConstantTrue
        } else {
            spirv::Op::ConstantFalse
        };
        let inst = dr::Instruction::new(opcode, Some(type_id), Some(result_id), vec![]);
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant_composite(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantComposite missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantComposite missing result id");
            return;
        };
        if instruction.operands().is_empty() {
            self.module_builder.emit_error(
                opcode_pos,
                "OpConstantComposite requires at least one constituent",
            );
            return;
        }
        let mut constituents = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::Id(id) => constituents.push(self.module_builder.resolve_id_ref(*id)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Constituent must be an id reference",
                    );
                    return;
                }
            }
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let operands = constituents.into_iter().map(dr::Operand::IdRef).collect();
        let inst = dr::Instruction::new(
            spirv::Op::ConstantComposite,
            Some(type_id),
            Some(result_id),
            operands,
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_constant_null(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull missing result id");
            return;
        };
        if !instruction.operands().is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpConstantNull does not take operands");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            spirv::Op::ConstantNull,
            Some(type_id),
            Some(result_id),
            vec![],
        );
        self.builder.module_mut().types_global_values.push(inst);
        self.record_from_module(|module| module.types_global_values.last().cloned());
    }

    fn translate_composite_extract(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(composite_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract missing composite value");
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite value") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    indexes.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Composite extract indexes must be literals",
                    );
                    return;
                }
            }
        }
        if indexes.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeExtract requires at least one index");
            return;
        }
        let mismatch_position = indexes
            .first()
            .map(|(_, span)| *span)
            .unwrap_or_else(|| composite_operand.span().start());
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(composite_type_id) = self.module_builder.value_type(composite_id) else {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Composite operand has no known type",
            );
            return;
        };
        let Some(expected_type) = self.resolve_composite_access(composite_type_id, &indexes) else {
            return;
        };
        if expected_type != type_id {
            self.module_builder.emit_error(
                mismatch_position,
                "Result type does not match the selected component type",
            );
            return;
        }
        let literal_indexes: Vec<u32> = indexes.iter().map(|(value, _)| *value).collect();
        match self.builder.composite_extract(
            type_id,
            Some(result_id),
            composite_id,
            literal_indexes,
        ) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_composite_insert(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(object_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing object operand");
            return;
        };
        let Some(object_id) = self.operand_as_id(object_operand, "object operand") else {
            return;
        };
        let Some(composite_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert missing composite operand");
            return;
        };
        let Some(composite_id) = self.operand_as_id(composite_operand, "composite operand") else {
            return;
        };
        let mut indexes = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => {
                    indexes.push((literal_to_u32(literal), operand.span().start()))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Composite insert indexes must be literals",
                    );
                    return;
                }
            }
        }
        if indexes.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpCompositeInsert requires at least one index");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let Some(composite_type_id) = self.module_builder.value_type(composite_id) else {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Composite operand has no known type",
            );
            return;
        };
        if composite_type_id != type_id {
            self.module_builder.emit_error(
                composite_operand.span().start(),
                "Result type must match the composite operand type",
            );
            return;
        }
        let Some(target_type) = self.resolve_composite_access(composite_type_id, &indexes) else {
            return;
        };
        let Some(object_type_id) = self.module_builder.value_type(object_id) else {
            self.module_builder.emit_error(
                object_operand.span().start(),
                "Object operand has no known type",
            );
            return;
        };
        if object_type_id != target_type {
            self.module_builder.emit_error(
                object_operand.span().start(),
                "Object operand type must match the selected component type",
            );
            return;
        }
        let literal_indexes: Vec<u32> = indexes.iter().map(|(value, _)| *value).collect();
        match self.builder.composite_insert(
            type_id,
            Some(result_id),
            object_id,
            composite_id,
            literal_indexes,
        ) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_ext_inst(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(set_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing instruction set");
            return;
        };
        let Some(set_id) = self.operand_as_id(set_operand, "instruction set") else {
            return;
        };
        let Some(import_info) = self.module_builder.ext_inst_import(set_id).cloned() else {
            self.module_builder.emit_error(
                set_operand.span().start(),
                "OpExtInst references an unknown instruction set",
            );
            return;
        };
        let Some(opcode_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpExtInst missing opcode operand");
            return;
        };
        let Some(resolved) = self.resolve_ext_inst_opcode(&import_info, opcode_operand) else {
            return;
        };
        let mut encoded_operands = vec![
            dr::Operand::IdRef(set_id),
            dr::Operand::LiteralExtInstInteger(resolved.opcode),
        ];
        if let Some(descriptors) = resolved.operands {
            for descriptor in descriptors {
                match descriptor.quantifier {
                    OperandQuantifier::One => {
                        let Some(next_operand) = operands.next() else {
                            self.module_builder
                                .emit_error(opcode_pos, "Extended instruction missing operands");
                            return;
                        };
                        let Some(encoded) =
                            self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                        else {
                            return;
                        };
                        encoded_operands.push(encoded);
                    }
                    OperandQuantifier::ZeroOrOne => {
                        if operands.as_slice().is_empty() {
                            continue;
                        }
                        let next_operand = operands.next().expect("peeked operand");
                        let Some(encoded) =
                            self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                        else {
                            return;
                        };
                        encoded_operands.push(encoded);
                    }
                    OperandQuantifier::ZeroOrMore => {
                        for next_operand in operands.by_ref() {
                            let Some(encoded) =
                                self.encode_ext_inst_operand(descriptor, next_operand, opcode_pos)
                            else {
                                return;
                            };
                            encoded_operands.push(encoded);
                        }
                    }
                }
            }
            if let Some(extra) = operands.next() {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "Extended instruction received unexpected operands",
                );
                return;
            }
        } else {
            for operand in operands {
                let Some(encoded) = self.encode_generic_ext_inst_operand(operand) else {
                    return;
                };
                encoded_operands.push(encoded);
            }
        }

        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            spirv::Op::ExtInst,
            Some(type_id),
            Some(result_id),
            encoded_operands,
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_decorate(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(target_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpDecorate missing target id");
            return;
        };
        let Some(decoration_operand) = instruction.operands().get(1) else {
            self.module_builder
                .emit_error(opcode_pos, "OpDecorate missing decoration enumerant");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "decorated id") else {
            return;
        };
        let Some(decoration) = self.parse_enum_operand::<spirv::Decoration>(
            Some(decoration_operand),
            "decoration",
            opcode_pos,
        ) else {
            return;
        };
        let mut operands = vec![
            dr::Operand::IdRef(target_id),
            dr::Operand::Decoration(decoration),
        ];
        if let Some(extra) =
            self.encode_decoration_operands(decoration, instruction.operands(), 2, opcode_pos)
        {
            operands.extend(extra);
        } else {
            return;
        }
        self.push_annotation_instruction(dr::Instruction::new(
            spirv::Op::Decorate,
            None,
            None,
            operands,
        ));
    }

    fn translate_decorate_id(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_decorate(instruction);
    }

    fn translate_decorate_string(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_decorate(instruction);
    }

    fn translate_member_decorate(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(struct_operand) = instruction.operands().first() else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing struct id");
            return;
        };
        let Some(member_operand) = instruction.operands().get(1) else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing member index");
            return;
        };
        let Some(decoration_operand) = instruction.operands().get(2) else {
            self.module_builder
                .emit_error(opcode_pos, "OpMemberDecorate missing decoration enumerant");
            return;
        };
        let Some(struct_id) = self.operand_as_id(struct_operand, "struct id") else {
            return;
        };
        let Some(member_index_value) = (match member_operand.value() {
            OperandValue::Literal(literal) => Some(literal_to_u32(literal)),
            _ => {
                self.module_builder.emit_error(
                    member_operand.span().start(),
                    "Member index must be a literal",
                );
                None
            }
        }) else {
            return;
        };
        let Ok(member_index) = usize::try_from(member_index_value) else {
            self.module_builder
                .emit_error(member_operand.span().start(), "Member index is too large");
            return;
        };
        let Some(decoration) = self.parse_enum_operand::<spirv::Decoration>(
            Some(decoration_operand),
            "decoration",
            opcode_pos,
        ) else {
            return;
        };
        let mut operands = vec![
            dr::Operand::IdRef(struct_id),
            dr::Operand::LiteralBit32(member_index_value),
            dr::Operand::Decoration(decoration),
        ];
        if let Some(extra) =
            self.encode_decoration_operands(decoration, instruction.operands(), 3, opcode_pos)
        {
            operands.extend(extra);
        } else {
            return;
        }
        self.apply_member_decorate_metadata(struct_id, member_index, decoration, instruction);
        self.push_annotation_instruction(dr::Instruction::new(
            spirv::Op::MemberDecorate,
            None,
            None,
            operands,
        ));
    }

    fn translate_member_decorate_string(&mut self, instruction: &ParsedInstruction<'a>) {
        self.translate_member_decorate(instruction);
    }

    fn apply_member_decorate_metadata(
        &mut self,
        struct_id: u32,
        member_index: usize,
        decoration: spirv::Decoration,
        instruction: &ParsedInstruction<'a>,
    ) {
        match decoration {
            spirv::Decoration::RowMajor => {
                let position = instruction
                    .operands()
                    .get(2)
                    .map(|operand| operand.span().start())
                    .unwrap_or_default();
                self.module_builder.apply_member_majorness(
                    struct_id,
                    member_index,
                    MatrixMajorness::RowMajor,
                    position,
                );
            }
            spirv::Decoration::ColMajor => {
                let position = instruction
                    .operands()
                    .get(2)
                    .map(|operand| operand.span().start())
                    .unwrap_or_default();
                self.module_builder.apply_member_majorness(
                    struct_id,
                    member_index,
                    MatrixMajorness::ColumnMajor,
                    position,
                );
            }
            spirv::Decoration::MatrixStride => {
                let Some(stride_operand) = instruction.operands().get(3) else {
                    return;
                };
                let stride = match stride_operand.value() {
                    OperandValue::Literal(literal) => literal_to_u32(literal),
                    _ => {
                        self.module_builder.emit_error(
                            stride_operand.span().start(),
                            "MatrixStride requires an integer literal",
                        );
                        return;
                    }
                };
                self.module_builder.apply_member_matrix_stride(
                    struct_id,
                    member_index,
                    stride,
                    stride_operand.span().start(),
                );
            }
            _ => {}
        }
    }

    fn translate_phi(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi missing result id");
            return;
        };
        let mut incoming = Vec::new();
        for operand in instruction.operands() {
            match operand.value() {
                OperandValue::IdPair(value, parent) => {
                    let value_id = self.module_builder.resolve_id_ref(*value);
                    let parent_id = self.module_builder.resolve_id_ref(*parent);
                    incoming.push((value_id, parent_id));
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "OpPhi operands must be id pairs");
                    return;
                }
            }
        }
        if incoming.is_empty() {
            self.module_builder
                .emit_error(opcode_pos, "OpPhi requires incoming edges");
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        match self.builder.phi(type_id, Some(result_id), incoming) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_copy_memory(&mut self, instruction: &ParsedInstruction<'a>, sized: bool) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(target_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCopyMemory missing target");
            return;
        };
        let Some(source_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpCopyMemory missing source");
            return;
        };
        let Some(target_id) = self.operand_as_id(target_operand, "target pointer") else {
            return;
        };
        let Some(source_id) = self.operand_as_id(source_operand, "source pointer") else {
            return;
        };
        let mut dr_operands = vec![dr::Operand::IdRef(target_id), dr::Operand::IdRef(source_id)];
        if sized {
            let Some(size_operand) = operands.next() else {
                self.module_builder
                    .emit_error(opcode_pos, "OpCopyMemorySized missing size operand");
                return;
            };
            let Some(size_id) = self.operand_as_id(size_operand, "size operand") else {
                return;
            };
            dr_operands.push(dr::Operand::IdRef(size_id));
        }
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access masks",
            );
            return;
        }
        let opcode = if sized {
            spirv::Op::CopyMemorySized
        } else {
            spirv::Op::CopyMemory
        };
        let inst = dr::Instruction::new(opcode, None, None, dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_branch(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let operand = match instruction.operands().first() {
            Some(op) => op,
            None => {
                self.module_builder
                    .emit_error(opcode_pos, "OpBranch missing target");
                return;
            }
        };
        let Some(target) = self.operand_as_id(operand, "branch target") else {
            return;
        };
        match self.builder.branch(target) {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, operand.span().start()),
        }
    }

    fn translate_branch_conditional(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(condition_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing condition");
            return;
        };
        let Some(true_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing true label");
            return;
        };
        let Some(false_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpBranchConditional missing false label");
            return;
        };
        let Some(condition_id) = self.operand_as_id(condition_operand, "condition id") else {
            return;
        };
        let Some(true_label) = self.operand_as_id(true_operand, "true label") else {
            return;
        };
        let Some(false_label) = self.operand_as_id(false_operand, "false label") else {
            return;
        };
        let mut branch_weights = Vec::new();
        for operand in operands {
            match operand.value() {
                OperandValue::Literal(literal) => branch_weights.push(literal_to_u32(literal)),
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Branch weights must be literal integers",
                    );
                    return;
                }
            }
        }
        match self
            .builder
            .branch_conditional(condition_id, true_label, false_label, branch_weights)
        {
            Ok(_) => self.record_from_current_block(),
            Err(error) => self.emit_builder_error(error, opcode_pos),
        }
    }

    fn translate_variable(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(storage_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpVariable missing storage class");
            return;
        };
        let Some(storage_class) = self.parse_enum_operand::<spirv::StorageClass>(
            Some(storage_operand),
            "storage class",
            opcode_pos,
        ) else {
            return;
        };
        let initializer = match operands.next() {
            Some(operand) => match self.operand_as_id(operand, "initializer") {
                Some(id) => Some(id),
                None => return,
            },
            None => None,
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "OpVariable received unexpected operands",
            );
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let initializer_id = initializer;
        self.builder
            .variable(type_id, Some(result_id), storage_class, initializer_id);
        match (
            self.builder.selected_function(),
            self.builder.selected_block(),
        ) {
            (Some(_), Some(_)) => self.record_from_current_block(),
            _ => self.record_from_module(|module| module.types_global_values.last().cloned()),
        }
    }

    fn translate_load(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpLoad missing pointer operand");
            return;
        };
        let Some(pointer_id) = self.operand_as_id(pointer_operand, "pointer operand") else {
            return;
        };
        let mut dr_operands = vec![dr::Operand::IdRef(pointer_id)];
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access mask",
            );
            return;
        }
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst =
            dr::Instruction::new(spirv::Op::Load, Some(type_id), Some(result_id), dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_store(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let mut operands = instruction.operands().iter();
        let Some(pointer_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpStore missing pointer operand");
            return;
        };
        let Some(object_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "OpStore missing object operand");
            return;
        };
        let Some(pointer_id) = self.operand_as_id(pointer_operand, "pointer operand") else {
            return;
        };
        let Some(object_id) = self.operand_as_id(object_operand, "object operand") else {
            return;
        };
        let mut dr_operands = vec![
            dr::Operand::IdRef(pointer_id),
            dr::Operand::IdRef(object_id),
        ];
        self.take_memory_access_operand(&mut operands, &mut dr_operands);
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unexpected operands after memory access mask",
            );
            return;
        }
        let inst = dr::Instruction::new(spirv::Op::Store, None, None, dr_operands);
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_binary_arithmetic(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(lhs_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation missing operands");
            return;
        };
        let Some(rhs_operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Binary operation requires two operands");
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Binary operation received unexpected operands",
            );
            return;
        }
        let Some(lhs_id) = self.operand_as_id(lhs_operand, "left operand") else {
            return;
        };
        let Some(rhs_id) = self.operand_as_id(rhs_operand, "right operand") else {
            return;
        };
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::IdRef(lhs_id), dr::Operand::IdRef(rhs_id)],
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn translate_unary_op(&mut self, instruction: &ParsedInstruction<'a>) {
        let opcode_pos = instruction.opcode_position();
        let Some(result_type) = instruction.result_type() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing result type");
            return;
        };
        let Some(result_id) = instruction.result_id() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing result id");
            return;
        };
        let mut operands = instruction.operands().iter();
        let Some(operand) = operands.next() else {
            self.module_builder
                .emit_error(opcode_pos, "Unary operation missing operand");
            return;
        };
        if let Some(extra) = operands.next() {
            self.module_builder.emit_error(
                extra.span().start(),
                "Unary operation received unexpected operands",
            );
            return;
        }
        let Some(operand_id) = self.operand_as_id(operand, "operand") else {
            return;
        };
        let (type_id, result_id) = self
            .module_builder
            .bind_typed_result(result_type, result_id);
        let inst = dr::Instruction::new(
            instruction.opcode(),
            Some(type_id),
            Some(result_id),
            vec![dr::Operand::IdRef(operand_id)],
        );
        self.push_block_instruction(inst, opcode_pos);
    }

    fn operand_as_id(&mut self, operand: &ParsedOperand<'a>, label: &str) -> Option<u32> {
        match operand.value() {
            OperandValue::Id(id) => Some(self.module_builder.resolve_id_ref(*id)),
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("{label} must be an id reference"),
                );
                None
            }
        }
    }

    fn push_block_instruction(&mut self, instruction: dr::Instruction, position: MessagePosition) {
        match self
            .builder
            .insert_into_block(InsertPoint::End, instruction)
        {
            Ok(()) => self.record_from_current_block(),
            Err(BuildError::DetachedInstruction(Some(inst))) => {
                self.builder.module_mut().types_global_values.push(inst);
                self.record_from_module(|module| module.types_global_values.last().cloned());
            }
            Err(error) => self.emit_builder_error(error, position),
        }
    }

    fn push_annotation_instruction(&mut self, instruction: dr::Instruction) {
        self.builder.module_mut().annotations.push(instruction);
        self.record_from_module(|module| module.annotations.last().cloned());
    }

    fn take_memory_access_operand(
        &mut self,
        operands: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
        target: &mut Vec<dr::Operand>,
    ) {
        if let Some(next) = operands.as_slice().first() {
            if next.descriptor().kind() == OperandKind::MemoryAccess {
                let operand = operands.next().expect("peeked operand");
                self.encode_memory_access_operand(operand, target);
            }
        }
    }

    fn encode_memory_access_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
        target: &mut Vec<dr::Operand>,
    ) {
        let OperandValue::MemoryAccess(memory) = operand.value() else {
            self.module_builder
                .emit_error(operand.span().start(), "Invalid memory access operand");
            return;
        };
        target.push(dr::Operand::MemoryAccess(memory.mask()));
        if let Some(alignment) = memory.alignment() {
            target.push(dr::Operand::LiteralBit32(literal_to_u32(alignment)));
        }
        if let Some(scope) = memory.make_pointer_available_scope() {
            target.push(dr::Operand::IdRef(
                self.module_builder.resolve_id_ref(scope),
            ));
        }
        if let Some(scope) = memory.make_pointer_visible_scope() {
            target.push(dr::Operand::IdRef(
                self.module_builder.resolve_id_ref(scope),
            ));
        }
    }

    fn encode_decoration_operands(
        &mut self,
        decoration: spirv::Decoration,
        operands: &[ParsedOperand<'a>],
        start_index: usize,
        opcode_pos: MessagePosition,
    ) -> Option<Vec<dr::Operand>> {
        let mut encoded = Vec::new();
        let mut index = start_index;
        let descriptors = decoration_operand_descriptors(decoration);
        for descriptor in descriptors {
            match descriptor.quantifier() {
                OperandQuantifier::One => {
                    let Some(next) = operands.get(index) else {
                        self.module_builder
                            .emit_error(opcode_pos, "Decoration missing required operand");
                        return None;
                    };
                    let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                    encoded.push(value);
                    index += 1;
                }
                OperandQuantifier::ZeroOrOne => {
                    if let Some(next) = operands.get(index) {
                        let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                        encoded.push(value);
                        index += 1;
                    }
                }
                OperandQuantifier::ZeroOrMore => {
                    while let Some(next) = operands.get(index) {
                        let value = self.encode_operand_for_kind(next, descriptor.kind())?;
                        encoded.push(value);
                        index += 1;
                    }
                }
            }
        }
        if index != operands.len() {
            if let Some(extra) = operands.get(index) {
                self.module_builder.emit_error(
                    extra.span().start(),
                    "Decoration received unexpected operands",
                );
            }
            return None;
        }
        Some(encoded)
    }

    fn encode_operand_for_kind(
        &mut self,
        operand: &ParsedOperand<'a>,
        kind: OperandKind,
    ) -> Option<dr::Operand> {
        use rspirv::grammar::OperandKind::*;
        match kind {
            IdRef => self
                .operand_as_id(operand, "decoration id")
                .map(dr::Operand::IdRef),
            IdScope => self
                .operand_as_id(operand, "decoration scope id")
                .map(dr::Operand::IdScope),
            IdMemorySemantics => self
                .operand_as_id(operand, "decoration memory semantics id")
                .map(dr::Operand::IdMemorySemantics),
            LiteralInteger | LiteralContextDependentNumber => match operand.value() {
                OperandValue::Literal(literal) => Some(encode_literal_operand(literal)),
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Expected literal integer operand");
                    None
                }
            },
            LiteralString => match operand.value() {
                OperandValue::String(value) => {
                    Some(dr::Operand::LiteralString(parse_string_literal(value)))
                }
                _ => {
                    self.module_builder
                        .emit_error(operand.span().start(), "Expected string literal operand");
                    None
                }
            },
            LiteralFloat => self.encode_float_operand(operand),
            BuiltIn => self.encode_enumerant_operand::<spirv::BuiltIn>(operand, "built-in"),
            FunctionParameterAttribute => self
                .encode_enumerant_operand::<spirv::FunctionParameterAttribute>(
                    operand,
                    "function parameter attribute",
                ),
            FPRoundingMode => {
                self.encode_enumerant_operand::<spirv::FPRoundingMode>(operand, "FP rounding mode")
            }
            FPFastMathMode => self.encode_fp_fast_math_mode(operand),
            FPDenormMode => {
                self.encode_enumerant_operand::<spirv::FPDenormMode>(operand, "FP denorm mode")
            }
            FPOperationMode => self
                .encode_enumerant_operand::<spirv::FPOperationMode>(operand, "FP operation mode"),
            LinkageType => {
                self.encode_enumerant_operand::<spirv::LinkageType>(operand, "linkage type")
            }
            AccessQualifier => {
                self.encode_enumerant_operand::<spirv::AccessQualifier>(operand, "access qualifier")
            }
            HostAccessQualifier => self.encode_enumerant_operand::<spirv::HostAccessQualifier>(
                operand,
                "host access qualifier",
            ),
            InitializationModeQualifier => self
                .encode_enumerant_operand::<spirv::InitializationModeQualifier>(
                    operand,
                    "initialization mode",
                ),
            LoadCacheControl => {
                self.encode_enumerant_operand::<spirv::LoadCacheControl>(operand, "cache control")
            }
            StoreCacheControl => {
                self.encode_enumerant_operand::<spirv::StoreCacheControl>(operand, "cache control")
            }
            _ => {
                // Decoration operands should only reference the operand kinds enumerated in the
                // grammar subset we generated. If we reach here, the grammar introduced a new
                // operand kind that needs encoding support.
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("Unsupported decoration operand kind: {:?}", kind),
                );
                None
            }
        }
    }

    fn encode_float_operand(&mut self, operand: &ParsedOperand<'a>) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(encode_literal_operand(literal)),
            OperandValue::Word(word) => {
                let text = word.as_str();
                match text.parse::<f64>() {
                    Ok(value64) => {
                        if value64.is_nan() {
                            Some(dr::Operand::LiteralBit32(f32::NAN.to_bits()))
                        } else {
                            let value32 = value64 as f32;
                            if value64 == (value32 as f64) {
                                Some(dr::Operand::LiteralBit32(value32.to_bits()))
                            } else {
                                Some(dr::Operand::LiteralBit64(value64.to_bits()))
                            }
                        }
                    }
                    Err(_) => {
                        self.module_builder
                            .emit_error(operand.span().start(), "Failed to parse float literal");
                        None
                    }
                }
            }
            _ => {
                self.module_builder
                    .emit_error(operand.span().start(), "Float literal expected");
                None
            }
        }
    }

    fn encode_fp_fast_math_mode(&mut self, operand: &ParsedOperand<'a>) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(dr::Operand::FPFastMathMode(
                spirv::FPFastMathMode::from_bits_truncate(literal_to_u32(literal)),
            )),
            OperandValue::Word(word) => {
                let mut mode = spirv::FPFastMathMode::empty();
                for part in word.as_str().split('|').map(str::trim) {
                    if part.is_empty() || part == "None" {
                        continue;
                    }
                    let flag = match part {
                        "NotNaN" => spirv::FPFastMathMode::NOT_NAN,
                        "NotInf" => spirv::FPFastMathMode::NOT_INF,
                        "NSZ" => spirv::FPFastMathMode::NSZ,
                        "AllowRecip" => spirv::FPFastMathMode::ALLOW_RECIP,
                        "Fast" => spirv::FPFastMathMode::FAST,
                        "AllowContract" | "AllowContractFastINTEL" => {
                            spirv::FPFastMathMode::ALLOW_CONTRACT
                        }
                        "AllowReassoc" | "AllowReassocINTEL" => {
                            spirv::FPFastMathMode::ALLOW_REASSOC
                        }
                        "AllowTransform" | "AllowTransformINTEL" => {
                            spirv::FPFastMathMode::ALLOW_TRANSFORM
                        }
                        other => {
                            self.module_builder.emit_error(
                                operand.span().start(),
                                format!("Unknown FPFastMathMode flag '{other}'"),
                            );
                            return None;
                        }
                    };
                    mode |= flag;
                }
                Some(dr::Operand::FPFastMathMode(mode))
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "FPFastMathMode operand must be literal or enumerant",
                );
                None
            }
        }
    }

    fn parse_fp_encoding_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::FPEncoding> {
        match operand.value() {
            OperandValue::Literal(literal) => {
                let value = literal_to_u32(literal);
                match spirv::FPEncoding::from_u32(value) {
                    Some(encoding) => Some(encoding),
                    None => {
                        self.module_builder.emit_error(
                            operand.span().start(),
                            format!("Unknown FPEncoding literal {value}"),
                        );
                        None
                    }
                }
            }
            OperandValue::Word(word) => match word.as_str().parse::<spirv::FPEncoding>() {
                Ok(encoding) => Some(encoding),
                Err(_) => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown FPEncoding '{}'", word.as_str()),
                    );
                    None
                }
            },
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "FPEncoding operand must be literal or enumerant",
                );
                None
            }
        }
    }

    fn encode_enumerant_operand<E>(
        &mut self,
        operand: &ParsedOperand<'a>,
        label: &str,
    ) -> Option<dr::Operand>
    where
        E: FromStr,
        dr::Operand: From<E>,
    {
        self.parse_enum_operand_value::<E>(operand, label)
            .map(dr::Operand::from)
    }

    fn parse_enum_operand_value<E>(&mut self, operand: &ParsedOperand<'a>, label: &str) -> Option<E>
    where
        E: FromStr,
    {
        self.parse_enum_operand::<E>(Some(operand), label, operand.span().start())
    }

    fn resolve_composite_access(
        &mut self,
        mut type_id: u32,
        indexes: &[(u32, MessagePosition)],
    ) -> Option<u32> {
        for (depth, (index, position)) in indexes.iter().enumerate() {
            let Some(info) = self.module_builder.composite_type(type_id) else {
                self.module_builder
                    .emit_error(*position, "Operand type is not a composite value");
                return None;
            };
            match info {
                CompositeTypeInfo::Vector(vector) => {
                    let vector = *vector;
                    if *index >= vector.component_count() {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Composite extract index {index} exceeds vector width {}",
                                vector.component_count()
                            ),
                        );
                        return None;
                    }
                    if depth + 1 != indexes.len() {
                        self.module_builder
                            .emit_error(*position, "Cannot descend past a vector component");
                        return None;
                    }
                    type_id = vector.component_type();
                }
                CompositeTypeInfo::Array(array) => {
                    let array = *array;
                    let Some(length) = self.module_builder.array_length(&array) else {
                        self.module_builder.emit_error(
                            *position,
                            "Array length must be defined by an integer constant",
                        );
                        return None;
                    };
                    if *index >= length {
                        self.module_builder.emit_error(
                            *position,
                            format!("Array index {index} exceeds array length {length}",),
                        );
                        return None;
                    }
                    type_id = array.element_type();
                }
                CompositeTypeInfo::Struct(struct_info) => {
                    let Ok(field_index) = usize::try_from(*index) else {
                        self.module_builder
                            .emit_error(*position, "Struct index exceeds implementation limits");
                        return None;
                    };
                    let Some(field_type) = struct_info.field_type(field_index) else {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Struct index {index} exceeds field count {}",
                                struct_info.field_count()
                            ),
                        );
                        return None;
                    };
                    type_id = field_type;
                }
                CompositeTypeInfo::Matrix(matrix) => {
                    let matrix = *matrix;
                    if *index >= matrix.column_count() {
                        self.module_builder.emit_error(
                            *position,
                            format!(
                                "Matrix column index {index} exceeds column count {}",
                                matrix.column_count()
                            ),
                        );
                        return None;
                    }
                    type_id = matrix.column_type();
                }
            }
        }
        Some(type_id)
    }

    fn resolve_ext_inst_opcode(
        &mut self,
        info: &ExtInstImportInfo,
        operand: &ParsedOperand<'a>,
    ) -> Option<ResolvedExtInst<'static>> {
        match operand.value() {
            OperandValue::Literal(literal) => Some(ResolvedExtInst {
                opcode: literal_to_u32(literal),
                operands: None,
            }),
            OperandValue::Word(word) => {
                if let Some(inst) = info.kind.lookup(word.as_str()) {
                    return Some(inst.into());
                }
                if let Some(mapped) = lookup_custom_ext_inst_opcode(&info.name, word.as_str()) {
                    return Some(mapped);
                }
                if let Ok(value) = word.as_str().parse::<u32>() {
                    return Some(ResolvedExtInst {
                        opcode: value,
                        operands: None,
                    });
                }
                if info.kind.has_grammar() {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!(
                            "Unknown {} extended instruction '{}'",
                            info.name,
                            word.as_str()
                        ),
                    );
                } else {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!(
                            "Instruction set '{}' does not support named opcodes; use numeric identifiers",
                            info.name
                        ),
                    );
                }
                None
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Extended instruction opcode must be a literal or enumerant name",
                );
                None
            }
        }
    }

    fn encode_ext_inst_operand(
        &mut self,
        descriptor: &LogicalOperand,
        operand: &ParsedOperand<'a>,
        opcode_pos: MessagePosition,
    ) -> Option<dr::Operand> {
        match descriptor.kind {
            OperandKind::IdRef => self
                .operand_as_id(operand, "extended instruction operand")
                .map(dr::Operand::IdRef),
            OperandKind::LiteralInteger
            | OperandKind::LiteralContextDependentNumber
            | OperandKind::LiteralExtInstInteger => match operand.value() {
                OperandValue::Literal(literal) => {
                    Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Extended instruction literal operand expected",
                    );
                    None
                }
            },
            OperandKind::FPRoundingMode => {
                let mode = self.parse_enum_operand::<spirv::FPRoundingMode>(
                    Some(operand),
                    "rounding mode",
                    opcode_pos,
                )?;
                Some(dr::Operand::FPRoundingMode(mode))
            }
            OperandKind::LiteralString => match operand.value() {
                OperandValue::String(value) => {
                    Some(dr::Operand::LiteralString(parse_string_literal(value)))
                }
                _ => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        "Extended instruction string operand expected",
                    );
                    None
                }
            },
            _ => {
                self.module_builder.emit_error(
                    opcode_pos,
                    format!(
                        "Extended instruction operand kind {:?} is not supported",
                        descriptor.kind
                    ),
                );
                None
            }
        }
    }

    fn encode_generic_ext_inst_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<dr::Operand> {
        match operand.value() {
            OperandValue::Id(id) => {
                Some(dr::Operand::IdRef(self.module_builder.resolve_id_ref(*id)))
            }
            OperandValue::Literal(literal) => {
                Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
            }
            OperandValue::String(value) => {
                Some(dr::Operand::LiteralString(parse_string_literal(value)))
            }
            OperandValue::Word(word) => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!(
                        "Named opcode '{}' is not supported for this instruction set",
                        word.as_str()
                    ),
                );
                None
            }
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Extended instruction operand must be an id or literal",
                );
                None
            }
        }
    }

    fn parse_enum_operand<E>(
        &mut self,
        operand: Option<&ParsedOperand<'a>>,
        label: &str,
        missing_position: MessagePosition,
    ) -> Option<E>
    where
        E: FromStr,
    {
        let operand = match operand {
            Some(value) => value,
            None => {
                self.module_builder
                    .emit_error(missing_position, format!("Missing {label}"));
                return None;
            }
        };
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("{label} must be an enumerant"),
                );
                return None;
            }
        };
        match word.as_str().parse::<E>() {
            Ok(value) => Some(value),
            Err(_err) => {
                self.module_builder
                    .emit_error(operand.span().start(), format!("Invalid {label}"));
                None
            }
        }
    }

    fn parse_loop_control_operand(
        &mut self,
        operand: &ParsedOperand<'a>,
        remaining: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
    ) -> Option<(spirv::LoopControl, Vec<dr::Operand>)> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder
                    .emit_error(operand.span().start(), "Loop control must be an enumerant");
                return None;
            }
        };
        let mut control = spirv::LoopControl::empty();
        let mut additional_operands = Vec::new();
        for part in word.as_str().split('|').map(str::trim) {
            if part.is_empty() || part == "None" {
                continue;
            }
            match part {
                "Unroll" => control |= spirv::LoopControl::UNROLL,
                "DontUnroll" => control |= spirv::LoopControl::DONT_UNROLL,
                "DependencyInfinite" => control |= spirv::LoopControl::DEPENDENCY_INFINITE,
                "DependencyLength" => {
                    control |= spirv::LoopControl::DEPENDENCY_LENGTH;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "MinIterations" => {
                    control |= spirv::LoopControl::MIN_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "MaxIterations" => {
                    control |= spirv::LoopControl::MAX_ITERATIONS;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "IterationMultiple" => {
                    control |= spirv::LoopControl::ITERATION_MULTIPLE;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "PeelCount" => {
                    control |= spirv::LoopControl::PEEL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                "PartialCount" => {
                    control |= spirv::LoopControl::PARTIAL_COUNT;
                    additional_operands.push(self.expect_loop_control_literal(
                        remaining,
                        operand.span().start(),
                        part,
                    )?);
                }
                other => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown loop control flag '{other}'"),
                    );
                    return None;
                }
            }
        }
        Some((control, additional_operands))
    }

    fn expect_loop_control_literal(
        &mut self,
        remaining: &mut std::slice::Iter<'_, ParsedOperand<'a>>,
        flag_position: MessagePosition,
        label: &str,
    ) -> Option<dr::Operand> {
        let Some(operand) = remaining.next() else {
            self.module_builder.emit_error(
                flag_position,
                format!("Loop control flag {label} requires a literal operand"),
            );
            return None;
        };
        let literal = match operand.value() {
            OperandValue::Literal(lit) => lit,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    format!("Loop control flag {label} requires a literal operand"),
                );
                return None;
            }
        };
        Some(dr::Operand::LiteralBit32(literal_to_u32(literal)))
    }

    fn parse_function_control(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::FunctionControl> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Function control must be an enumerant",
                );
                return None;
            }
        };
        let text = word.as_str();
        if text == "None" {
            return Some(spirv::FunctionControl::empty());
        }

        let mut control = spirv::FunctionControl::empty();
        for part in text.split('|').map(str::trim) {
            if part.is_empty() {
                continue;
            }
            let flag = match part {
                "Inline" => Some(spirv::FunctionControl::INLINE),
                "DontInline" => Some(spirv::FunctionControl::DONT_INLINE),
                "Pure" => Some(spirv::FunctionControl::PURE),
                "Const" => Some(spirv::FunctionControl::CONST),
                _ => None,
            };
            match flag {
                Some(value) => control |= value,
                None => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown function control flag '{part}'"),
                    );
                    return None;
                }
            }
        }
        Some(control)
    }

    fn parse_selection_control(
        &mut self,
        operand: &ParsedOperand<'a>,
    ) -> Option<spirv::SelectionControl> {
        let word = match operand.value() {
            OperandValue::Word(word) => word,
            _ => {
                self.module_builder.emit_error(
                    operand.span().start(),
                    "Selection control must be an enumerant",
                );
                return None;
            }
        };
        let mut control = spirv::SelectionControl::empty();
        for part in word.as_str().split('|').map(str::trim) {
            if part.is_empty() || part == "None" {
                continue;
            }
            match part {
                "Flatten" => control |= spirv::SelectionControl::FLATTEN,
                "DontFlatten" => control |= spirv::SelectionControl::DONT_FLATTEN,
                other => {
                    self.module_builder.emit_error(
                        operand.span().start(),
                        format!("Unknown selection control flag '{other}'"),
                    );
                    return None;
                }
            }
        }
        Some(control)
    }

    fn emit_builder_error(&mut self, error: BuildError, position: MessagePosition) {
        self.module_builder
            .emit_error(position, format!("Assembler builder error: {error}"));
    }

    /// Finalizes the translation and returns the constructed module plus diagnostics.
    pub fn finish(self) -> (dr::Module, Vec<DiagnosticMessage<'static>>) {
        let output = self.into_parts();
        (output.module, output.diagnostics)
    }

    /// Finalizes the translation and returns the module, diagnostics, and span map.
    ///
    /// The span map is only populated if the translator was created with `track_spans: true`.
    pub fn finish_with_spans(
        self,
    ) -> (dr::Module, Vec<DiagnosticMessage<'static>>, Option<SpanMap>) {
        let output = self.into_parts();
        (output.module, output.diagnostics, output.span_map)
    }

    fn into_parts(self) -> AssemblyOutput {
        let (diagnostics, next_id, span_map) = self.module_builder.finish_with_spans();
        let mut module = self.builder.module();
        match module.header.as_mut() {
            Some(header) => header.bound = next_id,
            None => module.header = Some(dr::ModuleHeader::new(next_id)),
        }
        AssemblyOutput {
            module,
            diagnostics,
            span_map,
        }
    }
}

struct AssemblyOutput {
    module: dr::Module,
    diagnostics: Vec<DiagnosticMessage<'static>>,
    span_map: Option<SpanMap>,
}

fn configure_builder_for_env(builder: &mut dr::Builder, env: TargetEnv) {
    let version = env.spirv_version();
    builder.set_version(version.major(), version.minor());
}

/// Assembles a sequence of parsed instructions into a SPIR-V module, returning both the module and
/// any diagnostics emitted along the way.
pub fn assemble_instructions<'a>(
    instructions: &[&'a ParsedInstruction<'a>],
) -> Result<dr::Module, AssemblyError> {
    let mut translator = AssemblyTranslator::new();
    for instruction in instructions {
        translator.translate(instruction);
    }
    let AssemblyOutput {
        module,
        diagnostics,
        ..
    } = translator.into_parts();
    finalize_result(module, diagnostics)
}

fn literal_to_u32(literal: &LiteralNumber) -> u32 {
    match literal {
        LiteralNumber::Unsigned(value) => *value as u32,
        LiteralNumber::Signed(value) => *value as u32,
    }
}

fn literal_to_u64(literal: &LiteralNumber) -> u64 {
    match literal {
        LiteralNumber::Unsigned(value) => *value,
        LiteralNumber::Signed(value) => *value as u64,
    }
}

fn encode_literal_operand(literal: &LiteralNumber) -> dr::Operand {
    match literal {
        LiteralNumber::Unsigned(value) if *value <= u32::MAX as u64 => {
            dr::Operand::LiteralBit32(*value as u32)
        }
        LiteralNumber::Unsigned(value) => dr::Operand::LiteralBit64(*value),
        LiteralNumber::Signed(value) if *value >= i32::MIN as i64 && *value <= i32::MAX as i64 => {
            dr::Operand::LiteralBit32(*value as u32)
        }
        LiteralNumber::Signed(value) => dr::Operand::LiteralBit64(*value as u64),
    }
}

fn finalize_result<T>(
    value: T,
    diagnostics: Vec<DiagnosticMessage<'static>>,
) -> Result<T, AssemblyError> {
    if diagnostics.is_empty() {
        Ok(value)
    } else {
        Err(AssemblyError::new(diagnostics))
    }
}

/// Assembles a block of textual SPIR-V instructions separated by newlines into a binary module.
/// Returns the assembled words on success along with any diagnostics emitted along the way.
pub fn assemble_text(text: &str) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_translator(text, AssemblyTranslator::new())
}

/// Assembles SPIR-V text using the provided target environment to configure the module header.
pub fn assemble_text_with_env(text: &str, env: TargetEnv) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_options(text, env, TextToBinaryOptions::NONE)
}

/// Assembles SPIR-V text with the provided options and target environment.
pub fn assemble_text_with_options(
    text: &str,
    env: TargetEnv,
    options: TextToBinaryOptions,
) -> Result<Vec<u32>, AssemblyError> {
    assemble_text_with_translator(
        text,
        AssemblyTranslator::with_target_env_and_options(env, options),
    )
}

fn assemble_text_with_translator<'a>(
    text: &'a str,
    mut translator: AssemblyTranslator<'a>,
) -> Result<Vec<u32>, AssemblyError> {
    let mut diagnostics = Vec::new();
    let mut instructions = Vec::new();

    let mut line_bounds = Vec::new();
    let mut line_start = 0usize;
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            let mut line_end = idx;
            if line_end > line_start && text.as_bytes()[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            line_bounds.push((line_start, line_end));
            line_start = idx + 1;
        }
    }
    if line_start < text.len() {
        line_bounds.push((line_start, text.len()));
    }

    let mut line_index = 0usize;
    while line_index < line_bounds.len() {
        let (start, _) = line_bounds[line_index];
        let mut last_line = line_index;
        let mut span_end = line_bounds[last_line].1;
        loop {
            match process_line(text, start, span_end, line_index) {
                Ok(Some(parsed)) => {
                    instructions.push(parsed);
                    line_index = last_line + 1;
                    break;
                }
                Ok(None) => {
                    line_index = last_line + 1;
                    break;
                }
                Err(error) => {
                    let is_unterminated =
                        error.diagnostic().message() == "unterminated string literal";
                    if is_unterminated && last_line + 1 < line_bounds.len() {
                        last_line += 1;
                        span_end = line_bounds[last_line].1;
                        continue;
                    }
                    diagnostics.push(error.into_diagnostic());
                    line_index = last_line + 1;
                    break;
                }
            }
        }
    }

    translator.reserve_numeric_result_ids(&instructions);
    for instruction in &instructions {
        translator.translate(instruction);
    }

    let AssemblyOutput {
        module,
        diagnostics: mut translator_diagnostics,
        ..
    } = translator.into_parts();
    diagnostics.append(&mut translator_diagnostics);
    let words = match finalize_result((), diagnostics) {
        Ok(_) => module.assemble(),
        Err(error) => return Err(error),
    };
    Ok(words)
}

/// Result of assembling SPIR-V text with span tracking enabled.
#[derive(Debug)]
pub struct AssemblyWithSpans {
    /// The assembled SPIR-V binary words.
    pub words: Vec<u32>,
    /// Map from result IDs to their source locations.
    pub span_map: SpanMap,
}

/// Assembles SPIR-V text and tracks source locations for all result IDs.
///
/// This is useful for validation error reporting, as the span map can be
/// passed to the validator to provide precise source locations in errors.
///
/// # Example
///
/// ```ignore
/// use spirv_tools_core::assembly::assemble_text_with_spans;
///
/// let text = r#"
///     OpCapability Shader
///     OpMemoryModel Logical GLSL450
///     %void = OpTypeVoid
/// "#;
///
/// let result = assemble_text_with_spans(text)?;
/// // result.span_map now contains the source location for %void
/// ```
pub fn assemble_text_with_spans(text: &str) -> Result<AssemblyWithSpans, AssemblyError> {
    assemble_text_with_spans_and_env(text, TargetEnv::Universal1_6)
}

/// Assembles SPIR-V text with span tracking and a specific target environment.
pub fn assemble_text_with_spans_and_env(
    text: &str,
    env: TargetEnv,
) -> Result<AssemblyWithSpans, AssemblyError> {
    assemble_text_with_spans_full(text, env, TextToBinaryOptions::NONE)
}

/// Assembles SPIR-V text with span tracking, environment, and options.
pub fn assemble_text_with_spans_full(
    text: &str,
    env: TargetEnv,
    options: TextToBinaryOptions,
) -> Result<AssemblyWithSpans, AssemblyError> {
    let translator = AssemblyTranslator::with_full_options(env, options, true);
    assemble_text_with_translator_for_spans(text, translator)
}

fn assemble_text_with_translator_for_spans<'a>(
    text: &'a str,
    mut translator: AssemblyTranslator<'a>,
) -> Result<AssemblyWithSpans, AssemblyError> {
    let mut diagnostics = Vec::new();
    let mut instructions = Vec::new();

    let mut line_bounds = Vec::new();
    let mut line_start = 0usize;
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            let mut line_end = idx;
            if line_end > line_start && text.as_bytes()[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            line_bounds.push((line_start, line_end));
            line_start = idx + 1;
        }
    }
    if line_start < text.len() {
        line_bounds.push((line_start, text.len()));
    }

    let mut line_index = 0usize;
    while line_index < line_bounds.len() {
        let (start, _) = line_bounds[line_index];
        let mut last_line = line_index;
        let mut span_end = line_bounds[last_line].1;
        loop {
            match process_line(text, start, span_end, line_index) {
                Ok(Some(parsed)) => {
                    instructions.push(parsed);
                    line_index = last_line + 1;
                    break;
                }
                Ok(None) => {
                    line_index = last_line + 1;
                    break;
                }
                Err(error) => {
                    let is_unterminated =
                        error.diagnostic().message() == "unterminated string literal";
                    if is_unterminated && last_line + 1 < line_bounds.len() {
                        last_line += 1;
                        span_end = line_bounds[last_line].1;
                        continue;
                    }
                    diagnostics.push(error.into_diagnostic());
                    line_index = last_line + 1;
                    break;
                }
            }
        }
    }

    translator.reserve_numeric_result_ids(&instructions);
    for instruction in &instructions {
        translator.translate(instruction);
    }

    let AssemblyOutput {
        module,
        diagnostics: mut translator_diagnostics,
        span_map,
    } = translator.into_parts();
    diagnostics.append(&mut translator_diagnostics);

    match finalize_result((), diagnostics) {
        Ok(_) => Ok(AssemblyWithSpans {
            words: module.assemble(),
            span_map: span_map.unwrap_or_default(),
        }),
        Err(error) => Err(error),
    }
}

fn process_line<'a>(
    source: &'a str,
    line_start: usize,
    line_end: usize,
    line_index: usize,
) -> Result<Option<ParsedInstruction<'a>>, ParseError> {
    if line_start >= line_end {
        return Ok(None);
    }
    let line_slice = &source[line_start..line_end];
    let leading_ws = line_slice.len() - line_slice.trim_start().len();
    let trailing_ws = line_slice.len() - line_slice.trim_end().len();
    if leading_ws >= line_slice.len() - trailing_ws {
        return Ok(None);
    }
    let content_start = line_start + leading_ws;
    let content_end = line_end - trailing_ws;
    let line = &source[content_start..content_end];
    if line.is_empty() || line.starts_with(';') {
        return Ok(None);
    }
    let line_number = u32::try_from(line_index).unwrap_or(u32::MAX);
    let column_offset = u32::try_from(leading_ws).unwrap_or(u32::MAX);
    let index_offset = u32::try_from(content_start).unwrap_or(u32::MAX);
    let origin = MessagePosition::new(line_number, column_offset, index_offset);

    parse_instruction_with_origin(line, origin).map(Some)
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_instructions, assemble_text, assemble_text_with_env, assemble_text_with_options,
        AssemblyTranslator,
    };
    use crate::assembly::parser::parse_instruction;
    use crate::assembly::{BinaryToTextOptions, TextToBinaryOptions};
    use crate::disassembly::disassemble_binary;
    use crate::target_env::TargetEnv;
    use crate::version::SpirvVersion;
    use rspirv::{dr, spirv};

    #[test]
    fn translator_emits_type_int_instruction() {
        let parsed = parse_instruction("%uint = OpTypeInt 32 0").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module
            .types_global_values
            .first()
            .expect("type instruction");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(inst.result_id, Some(1));
        assert_eq!(inst.operands.len(), 2);
    }

    #[test]
    fn translator_sets_memory_model() {
        let parsed = parse_instruction("OpMemoryModel Logical GLSL450").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module.memory_model.as_ref().expect("memory model");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::MemoryModel);
    }

    #[test]
    fn translator_emits_entry_point_instruction() {
        let parsed =
            parse_instruction("OpEntryPoint GLCompute %main \"main\" %a %b").expect("parse");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module.entry_points.first().expect("entry point");
        assert_eq!(inst.class.opcode, rspirv::spirv::Op::EntryPoint);
    }

    #[test]
    fn translator_emits_extension_instruction() {
        let parsed =
            parse_instruction("OpExtension \"SPV_KHR_ray_tracing\"").expect("parse extension");
        let mut translator = AssemblyTranslator::new();
        translator.translate(&parsed);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let inst = module.extensions.first().expect("extension");
        assert_eq!(inst.class.opcode, spirv::Op::Extension);
        assert_eq!(
            inst.operands,
            vec![dr::Operand::LiteralString("SPV_KHR_ray_tracing".into())]
        );
    }

    #[test]
    fn assembler_preserves_textual_order_for_globals() {
        let input = r#"
; comment line
            OpMemoryModel Logical Simple
%glsl450 = OpExtInstImport "GLSL.std.450"
"#;
        let words = assemble_text(input).expect("assemble");
        assert!(
            words.len() >= 5,
            "assembled module should contain header and instructions"
        );
        let instructions = &words[5..];
        let memory_model = 196_622;
        let ext_inst_import = 393_227;
        let mem_idx = instructions
            .iter()
            .position(|word| *word == memory_model)
            .expect("memory model present");
        let ext_idx = instructions
            .iter()
            .position(|word| *word == ext_inst_import)
            .expect("ext inst import present");
        assert!(
            ext_idx < mem_idx,
            "assembler should canonicalize layout ordering (extinst before memory model)"
        );
    }

    #[test]
    fn arm_motion_engine_ext_inst_round_trips_with_names() {
        let src = [
            "%1 = OpExtInstImport \"Arm.MotionEngine.100\"",
            "%3 = OpExtInst %2 %1 MIN_SAD %4 %5 %6 %7 %8 %9 %10 %11 %12",
        ]
        .join("\n");
        let binary = assemble_text(&src).expect("assemble arm.motion");
        let disassembled =
            disassemble_binary(&binary, BinaryToTextOptions::NONE).expect("disassemble");
        assert!(
            disassembled.contains("MIN_SAD"),
            "expected disassembly to use the opcode name, got: {disassembled}"
        );
        assert!(
            !disassembled.contains(" OpExtInst %2 %1 0 "),
            "extinst opcode should not fall back to a numeric literal: {disassembled}"
        );
    }

    fn round_trip_with_options(
        text: &str,
        options: TextToBinaryOptions,
        disassemble_opts: BinaryToTextOptions,
    ) -> String {
        let binary =
            assemble_text_with_options(text, TargetEnv::Universal1_0, options).expect("assemble");
        disassemble_binary(&binary, disassemble_opts).expect("disassemble")
    }

    #[test]
    fn assembler_renumbers_numeric_ids_by_default() {
        let before = [
            "OpCapability Addresses",
            "OpCapability Kernel",
            "OpCapability GenericPointer",
            "OpCapability Linkage",
            "OpMemoryModel Physical32 OpenCL",
            "%i32 = OpTypeInt 32 1",
            "%u32 = OpTypeInt 32 0",
            "%f32 = OpTypeFloat 32",
            "%200 = OpTypeVoid",
            "%300 = OpTypeFunction %200",
            "%main = OpFunction %200 None %300",
            "%entry = OpLabel",
            "%100 = OpConstant %u32 100",
            "%1 = OpConstant %u32 200",
            "%2 = OpConstant %u32 300",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        let expected = [
            "OpCapability Addresses",
            "OpCapability Kernel",
            "OpCapability GenericPointer",
            "OpCapability Linkage",
            "OpMemoryModel Physical32 OpenCL",
            "%1 = OpTypeInt 32 1",
            "%2 = OpTypeInt 32 0",
            "%3 = OpTypeFloat 32",
            "%4 = OpTypeVoid",
            "%5 = OpTypeFunction %4",
            "%8 = OpConstant %2 100",
            "%9 = OpConstant %2 200",
            "%10 = OpConstant %2 300",
            "%6 = OpFunction %4 None %5",
            "%7 = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n")
            + "\n";

        let text = round_trip_with_options(
            &before,
            TextToBinaryOptions::NONE,
            BinaryToTextOptions::NO_HEADER,
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn assembler_preserves_numeric_ids_when_requested() {
        let before = [
            "OpCapability Addresses",
            "OpCapability Kernel",
            "OpCapability GenericPointer",
            "OpCapability Linkage",
            "OpMemoryModel Physical32 OpenCL",
            "%i32 = OpTypeInt 32 1",
            "%u32 = OpTypeInt 32 0",
            "%f32 = OpTypeFloat 32",
            "%200 = OpTypeVoid",
            "%300 = OpTypeFunction %200",
            "%main = OpFunction %200 None %300",
            "%entry = OpLabel",
            "%100 = OpConstant %u32 100",
            "%1 = OpConstant %u32 200",
            "%2 = OpConstant %u32 300",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");

        let expected = [
            "OpCapability Addresses",
            "OpCapability Kernel",
            "OpCapability GenericPointer",
            "OpCapability Linkage",
            "OpMemoryModel Physical32 OpenCL",
            "%3 = OpTypeInt 32 1",
            "%4 = OpTypeInt 32 0",
            "%5 = OpTypeFloat 32",
            "%200 = OpTypeVoid",
            "%300 = OpTypeFunction %200",
            "%100 = OpConstant %4 100",
            "%1 = OpConstant %4 200",
            "%2 = OpConstant %4 300",
            "%6 = OpFunction %200 None %300",
            "%7 = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n")
            + "\n";

        let text = round_trip_with_options(
            &before,
            TextToBinaryOptions::PRESERVE_NUMERIC_IDS,
            BinaryToTextOptions::NO_HEADER,
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn translator_emits_constant_instruction() {
        let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
        let const_inst = parse_instruction("%c32 = OpConstant %uint 32").unwrap();
        let mut translator = AssemblyTranslator::new();
        translator.translate(&type_inst);
        translator.translate(&const_inst);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        assert!(module
            .types_global_values
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Constant));
    }

    #[test]
    fn assemble_instructions_streams_sequence() {
        let type_inst = parse_instruction("%uint = OpTypeInt 32 0").unwrap();
        let mem_model = parse_instruction("OpMemoryModel Logical GLSL450").unwrap();
        let module =
            assemble_instructions(&[&type_inst, &mem_model]).expect("assemble instructions");
        assert!(module.memory_model.is_some());
    }

    #[test]
    fn assemble_text_parses_multiple_lines() {
        let text = "%uint = OpTypeInt 32 0\nOpMemoryModel Logical GLSL450";
        let binary = assemble_text(text).expect("assemble text");
        assert!(!binary.is_empty());
    }

    #[test]
    fn assemble_text_emits_simple_function() {
        let text = "\
%void = OpTypeVoid\n\
%void_fn = OpTypeFunction %void\n\
OpMemoryModel Logical GLSL450\n\
%main = OpFunction %void None %void_fn\n\
%entry = OpLabel\n\
OpReturn\n\
OpFunctionEnd";
        let binary = assemble_text(text).expect("assemble text");
        assert!(!binary.is_empty());
    }

    #[test]
    fn translator_handles_execution_mode_and_memory_ops() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint GLCompute %main \"main\" %buffer",
            "OpExecutionMode %main LocalSize 1 1 1",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr = OpTypePointer StorageBuffer %uint",
            "%one = OpConstant %uint 1",
            "%buffer = OpVariable %ptr StorageBuffer",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%value = OpLoad %uint %buffer",
            "%sum = OpIAdd %uint %value %one",
            "OpStore %buffer %sum",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| {
                parse_instruction(line)
                    .unwrap_or_else(|err| panic!("failed to parse '{line}': {err:?}"))
            })
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        assert_eq!(module.capabilities.len(), 1);
        assert_eq!(module.execution_modes.len(), 1);
        assert_eq!(module.entry_points.len(), 1);
        let function = module.functions.first().expect("function");
        assert_eq!(function.blocks.len(), 1);
        let block = function.blocks.first().expect("entry block");
        assert!(block
            .instructions
            .iter()
            .any(|inst| inst.class.opcode == spirv::Op::IAdd));
    }

    #[test]
    fn translator_emits_glsl_ext_inst_with_named_opcode() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%glsl = OpExtInstImport \"GLSL.std.450\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%zero = OpConstant %float 0",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%abs = OpExtInst %float %glsl FAbs %zero",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        assert_eq!(module.ext_inst_imports.len(), 1);
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let ext_inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
            .expect("ext inst instruction");
        assert!(matches!(
            ext_inst.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::LiteralExtInstInteger(4),
                dr::Operand::IdRef(_)
            ]
        ));
    }

    #[test]
    fn translator_emits_member_decorate_matrix_stride() {
        let source = [
            "%float = OpTypeFloat 32",
            "%vec4 = OpTypeVector %float 4",
            "%mat = OpTypeMatrix %vec4 4",
            "%struct = OpTypeStruct %mat",
            "OpMemberDecorate %struct 0 MatrixStride 16",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let struct_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::TypeStruct)
            .and_then(|inst| inst.result_id)
            .expect("struct id");
        let annotation = module.annotations.first().expect("annotation");
        assert_eq!(annotation.class.opcode, spirv::Op::MemberDecorate);
        assert_eq!(
            annotation.operands.as_slice(),
            [
                dr::Operand::IdRef(struct_id),
                dr::Operand::LiteralBit32(0),
                dr::Operand::Decoration(spirv::Decoration::MatrixStride),
                dr::Operand::LiteralBit32(16),
            ]
        );
    }

    #[test]
    fn diagnostics_report_original_positions() {
        let text = "OpCapability Shader\n    OpTypo Thing\n";
        let diagnostics = assemble_text(text)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert!(!diagnostics.is_empty());
        let position = diagnostics[0].position();
        assert_eq!(position.line(), 1);
        assert_eq!(position.column(), 4);
    }

    #[test]
    fn row_major_requires_matrix_stride() {
        let source = [
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat = OpTypeMatrix %vec2 2",
            "%struct = OpTypeStruct %mat",
            "OpMemberDecorate %struct 0 RowMajor",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "RowMajor decoration requires an accompanying MatrixStride"
        );
    }

    #[test]
    fn matrix_layout_requires_matrix_member_type() {
        let source = [
            "%float = OpTypeFloat 32",
            "%struct = OpTypeStruct %float",
            "OpMemberDecorate %struct 0 RowMajor",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "RowMajor decoration requires the member type to be a matrix or array of matrices"
        );
    }

    #[test]
    fn matrix_stride_requires_matrix_member() {
        let source = [
            "%float = OpTypeFloat 32",
            "%struct = OpTypeStruct %float",
            "OpMemberDecorate %struct 0 MatrixStride 16",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "MatrixStride decoration requires the member type to contain a matrix"
        );
    }

    #[test]
    fn conflicting_matrix_major_decorations_report_diagnostic() {
        let source = [
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat = OpTypeMatrix %vec2 2",
            "%struct = OpTypeStruct %mat",
            "OpMemberDecorate %struct 0 RowMajor",
            "OpMemberDecorate %struct 0 MatrixStride 16",
            "OpMemberDecorate %struct 0 ColMajor",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "RowMajor and ColMajor decorations cannot both target the same member"
        );
    }

    #[test]
    fn translator_emits_builtin_decorations() {
        let source = [
            "%float = OpTypeFloat 32",
            "%ptr = OpTypePointer Input %float",
            "%var = OpVariable %ptr Input",
            "OpDecorate %var BuiltIn Position",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let var_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Variable)
            .and_then(|inst| inst.result_id)
            .expect("variable id");
        let annotation = module.annotations.first().expect("annotation");
        assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
        assert_eq!(
            annotation.operands.as_slice(),
            [
                dr::Operand::IdRef(var_id),
                dr::Operand::Decoration(spirv::Decoration::BuiltIn),
                dr::Operand::BuiltIn(spirv::BuiltIn::Position),
            ]
        );
    }

    #[test]
    fn translator_emits_linkage_attributes() {
        let source = [
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
            "OpDecorate %main LinkageAttributes \"main\" Import",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function_id = module
            .functions
            .first()
            .and_then(|func| func.def.as_ref())
            .and_then(|inst| inst.result_id)
            .expect("function id");
        let annotation = module.annotations.first().expect("annotation");
        assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
        assert_eq!(
            annotation.operands.as_slice(),
            [
                dr::Operand::IdRef(function_id),
                dr::Operand::Decoration(spirv::Decoration::LinkageAttributes),
                dr::Operand::LiteralString("main".to_string()),
                dr::Operand::LinkageType(spirv::LinkageType::Import),
            ]
        );
    }

    #[test]
    fn translator_emits_decorate_id_operands() {
        let source = [
            "%uint = OpTypeInt 32 0",
            "%ptr = OpTypePointer Uniform %uint",
            "%var = OpVariable %ptr Uniform",
            "%const = OpConstant %uint 16",
            "OpDecorateId %var AlignmentId %const",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let var_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Variable)
            .and_then(|inst| inst.result_id)
            .expect("var id");
        let const_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Constant)
            .and_then(|inst| inst.result_id)
            .expect("const id");
        let annotation = module.annotations.first().expect("annotation");
        assert_eq!(annotation.class.opcode, spirv::Op::Decorate);
        assert_eq!(
            annotation.operands.as_slice(),
            [
                dr::Operand::IdRef(var_id),
                dr::Operand::Decoration(spirv::Decoration::AlignmentId),
                dr::Operand::IdRef(const_id),
            ]
        );
    }

    #[test]
    fn translator_handles_opencl_ext_inst_literal_operands() {
        let source = [
            "OpCapability Kernel",
            "OpMemoryModel Physical64 OpenCL",
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%ulong = OpTypeInt 64 0",
            "%ptr = OpTypePointer CrossWorkgroup %float",
            "%offset = OpConstant %ulong 1",
            "%addr = OpVariable %ptr CrossWorkgroup",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%load = OpExtInst %vec2 %opencl vloadn %offset %addr 2",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let ext_inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
            .expect("ext inst instruction");
        assert!(matches!(
            ext_inst.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::LiteralExtInstInteger(_),
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::LiteralBit32(2)
            ]
        ));
    }

    #[test]
    fn translator_handles_opencl_rounding_mode_operands() {
        let source = [
            "OpCapability Kernel",
            "OpMemoryModel Physical64 OpenCL",
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%ptr = OpTypePointer CrossWorkgroup %float",
            "%float_0 = OpConstant %float 0",
            "%value = OpConstantComposite %vec2 %float_0 %float_0",
            "%var = OpVariable %ptr CrossWorkgroup",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%call = OpExtInst %void %opencl vstore_half_r %value %var %value RTE",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let ext_inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
            .expect("ext inst instruction");
        assert!(matches!(
            ext_inst.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::LiteralExtInstInteger(_),
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::FPRoundingMode(spirv::FPRoundingMode::RTE)
            ]
        ));
    }

    #[test]
    fn translator_handles_opencl_printf_variadic_operands() {
        let source = [
            "OpCapability Kernel",
            "OpMemoryModel Physical64 OpenCL",
            "%opencl = OpExtInstImport \"OpenCL.std\"",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr = OpTypePointer CrossWorkgroup %uint",
            "%value = OpVariable %ptr CrossWorkgroup",
            "%format = OpVariable %ptr CrossWorkgroup",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%call = OpExtInst %void %opencl printf %format %value %value",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let ext_inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::ExtInst)
            .expect("ext inst instruction");
        assert_eq!(ext_inst.operands.len(), 5);
    }

    #[test]
    fn translator_emits_memory_operands_for_load_store() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr_ty = OpTypePointer StorageBuffer %uint",
            "%zero = OpConstant %uint 0",
            "%buffer = OpVariable %ptr_ty StorageBuffer",
            "%scope = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%val = OpLoad %uint %buffer Aligned|MakePointerVisible 8 %scope",
            "OpStore %buffer %val Aligned|MakePointerAvailable 8 %scope",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let load = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Load)
            .expect("load inst");
        assert!(matches!(
            load.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(mask),
                dr::Operand::LiteralBit32(8),
                dr::Operand::IdRef(_)
            ] if mask.contains(spirv::MemoryAccess::ALIGNED)
                && mask.contains(spirv::MemoryAccess::MAKE_POINTER_VISIBLE)
        ));
        let store = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Store)
            .expect("store inst");
        assert!(matches!(
            store.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(mask),
                dr::Operand::LiteralBit32(8),
                dr::Operand::IdRef(_)
            ] if mask.contains(spirv::MemoryAccess::ALIGNED)
                && mask.contains(spirv::MemoryAccess::MAKE_POINTER_AVAILABLE)
        ));
    }

    #[test]
    fn translator_emits_access_chain_instruction() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%ptr_uint = OpTypePointer StorageBuffer %uint",
            "%ptr_ptr_uint = OpTypePointer StorageBuffer %ptr_uint",
            "%zero = OpConstant %uint 0",
            "%var = OpVariable %ptr_ptr_uint StorageBuffer",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%elem_ptr = OpAccessChain %ptr_uint %var %zero",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let access_chain = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::AccessChain)
            .expect("access chain instruction");
        assert!(matches!(
            access_chain.operands.as_slice(),
            [dr::Operand::IdRef(_), dr::Operand::IdRef(_)]
        ));
    }

    #[test]
    fn translator_handles_branch_instructions() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpBranch %mid",
            "%mid = OpLabel",
            "OpBranchConditional %one %then %exit 1 2",
            "%then = OpLabel",
            "OpReturn",
            "%exit = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let all_insts: Vec<_> = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .collect();
        assert!(all_insts
            .iter()
            .any(|inst| inst.class.opcode == spirv::Op::Branch));
        let branch_cond = all_insts
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::BranchConditional)
            .expect("branch conditional inst");
        assert!(branch_cond.operands.len() >= 3);
    }

    #[test]
    fn translator_handles_copy_memory_operands() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%size = OpConstant %uint 4",
            "%ptr_fn = OpTypePointer Function %uint",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%dst = OpVariable %ptr_fn Function",
            "%src = OpVariable %ptr_fn Function",
            "OpCopyMemory %dst %src Aligned 4 Aligned 8",
            "OpCopyMemorySized %dst %src %size Aligned 4 Aligned 8",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let mut copies = block.instructions.iter().filter(|inst| {
            matches!(
                inst.class.opcode,
                spirv::Op::CopyMemory | spirv::Op::CopyMemorySized
            )
        });
        let copy = copies.next().expect("OpCopyMemory");
        assert!(matches!(
            copy.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(first),
                dr::Operand::LiteralBit32(4),
                dr::Operand::MemoryAccess(second),
                dr::Operand::LiteralBit32(8)
            ] if first.contains(spirv::MemoryAccess::ALIGNED)
                && second.contains(spirv::MemoryAccess::ALIGNED)
        ));
        let copy_sized = copies.next().expect("OpCopyMemorySized");
        assert!(matches!(
            copy_sized.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::MemoryAccess(first),
                dr::Operand::LiteralBit32(4),
                dr::Operand::MemoryAccess(second),
                dr::Operand::LiteralBit32(8)
            ] if first.contains(spirv::MemoryAccess::ALIGNED)
                && second.contains(spirv::MemoryAccess::ALIGNED)
        ));
    }

    #[test]
    fn translator_emits_selection_merge() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpSelectionMerge %merge None",
            "OpBranchConditional %one %then %else",
            "%then = OpLabel",
            "OpBranch %merge",
            "%else = OpLabel",
            "OpBranch %merge",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let selection_merge = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::SelectionMerge)
            .expect("selection merge");
        assert!(matches!(
            selection_merge.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::SelectionControl(control)
            ] if control.is_empty()
        ));
    }

    #[test]
    fn translator_emits_loop_merge_with_operands() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%uint = OpTypeInt 32 0",
            "%one = OpConstant %uint 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpBranch %loop",
            "%loop = OpLabel",
            "OpLoopMerge %merge %continue MinIterations|PartialCount 4 2",
            "OpBranch %continue",
            "%continue = OpLabel",
            "OpBranchConditional %one %loop %merge",
            "%merge = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let loop_merge = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::LoopMerge)
            .expect("loop merge");
        assert!(matches!(
            loop_merge.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::LoopControl(control),
                dr::Operand::LiteralBit32(4),
                dr::Operand::LiteralBit32(2)
            ] if control.contains(spirv::LoopControl::MIN_ITERATIONS)
                && control.contains(spirv::LoopControl::PARTIAL_COUNT)
        ));
    }

    #[test]
    fn translator_emits_composite_construct() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%uint_0 = OpConstant %int 0",
            "%uint_1 = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%vec = OpCompositeConstruct %vec2 %uint_0 %uint_1",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let inst = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::CompositeConstruct)
            .expect("composite construct");
        assert_eq!(inst.operands.len(), 2);
    }

    #[test]
    fn translator_emits_vector_shuffle() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%vec4 = OpTypeVector %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%two = OpConstant %int 2",
            "%three = OpConstant %int 3",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v1 = OpCompositeConstruct %vec2 %zero %one",
            "%v2 = OpCompositeConstruct %vec2 %two %three",
            "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 2 3",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let shuffle = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::VectorShuffle)
            .expect("vector shuffle");
        assert_eq!(shuffle.operands.len(), 6);
    }

    #[test]
    fn vector_shuffle_rejects_component_count_mismatch() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%vec4 = OpTypeVector %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v1 = OpCompositeConstruct %vec2 %zero %one",
            "%v2 = OpCompositeConstruct %vec2 %one %zero",
            "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 2",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "OpVectorShuffle expects 4 component literals but received 3"
        );
    }

    #[test]
    fn vector_shuffle_rejects_out_of_bounds_component() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%vec4 = OpTypeVector %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v1 = OpCompositeConstruct %vec2 %zero %one",
            "%v2 = OpCompositeConstruct %vec2 %one %zero",
            "%shuffle = OpVectorShuffle %vec4 %v1 %v2 0 1 5 3",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "Shuffle component 5 exceeds the available inputs (4)"
        );
    }

    #[test]
    fn translator_emits_composite_extract_and_insert() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v = OpCompositeConstruct %vec2 %zero %one",
            "%elem = OpCompositeExtract %int %v 1",
            "%result = OpCompositeInsert %vec2 %elem %v 0",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let mut extract_seen = false;
        let mut insert_seen = false;
        for inst in function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
        {
            if inst.class.opcode == spirv::Op::CompositeExtract {
                extract_seen = true;
                assert!(matches!(
                    inst.operands.as_slice(),
                    [dr::Operand::IdRef(_), dr::Operand::LiteralBit32(1)]
                ));
            }
            if inst.class.opcode == spirv::Op::CompositeInsert {
                insert_seen = true;
                assert!(matches!(
                    inst.operands.as_slice(),
                    [
                        dr::Operand::IdRef(_),
                        dr::Operand::IdRef(_),
                        dr::Operand::LiteralBit32(0)
                    ]
                ));
            }
        }
        assert!(extract_seen && insert_seen);
    }

    #[test]
    fn translator_handles_array_composite_extract() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%four = OpConstant %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%two = OpConstant %int 2",
            "%three = OpConstant %int 3",
            "%arr = OpTypeArray %int %four",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%value = OpCompositeConstruct %arr %zero %one %two %three",
            "%elem = OpCompositeExtract %int %value 2",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let extract = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::CompositeExtract)
            .expect("extract instruction");
        assert!(matches!(
            extract.operands.as_slice(),
            [dr::Operand::IdRef(_), dr::Operand::LiteralBit32(2)]
        ));
    }

    #[test]
    fn translator_handles_struct_composite_insert() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%two = OpConstant %int 2",
            "%struct = OpTypeStruct %int %int",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%value = OpCompositeConstruct %struct %zero %one",
            "%result = OpCompositeInsert %struct %two %value 1",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let insert = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::CompositeInsert)
            .expect("insert instruction");
        assert!(matches!(
            insert.operands.as_slice(),
            [
                dr::Operand::IdRef(_),
                dr::Operand::IdRef(_),
                dr::Operand::LiteralBit32(1)
            ]
        ));
    }

    #[test]
    fn translator_handles_matrix_composite_extract() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "%float_0 = OpConstant %float 0",
            "%float_1 = OpConstant %float 1",
            "%float_2 = OpConstant %float 2",
            "%float_3 = OpConstant %float 3",
            "%col0 = OpConstantComposite %vec2 %float_0 %float_1",
            "%col1 = OpConstantComposite %vec2 %float_2 %float_3",
            "%mat = OpConstantComposite %mat2 %col0 %col1",
            "%struct = OpTypeStruct %mat2",
            "%value = OpConstantComposite %struct %mat",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%elem = OpCompositeExtract %float %value 0 1 0",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("block");
        let extract = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::CompositeExtract)
            .expect("extract instruction");
        assert_eq!(extract.operands.len(), 4);
    }

    #[test]
    fn translator_handles_nested_composite_extract() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%uint = OpTypeInt 32 0",
            "%two = OpConstant %uint 2",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "%arr = OpTypeArray %mat2 %two",
            "%struct = OpTypeStruct %arr",
            "%f0 = OpConstant %float 0",
            "%f1 = OpConstant %float 1",
            "%f2 = OpConstant %float 2",
            "%f3 = OpConstant %float 3",
            "%f4 = OpConstant %float 4",
            "%f5 = OpConstant %float 5",
            "%f6 = OpConstant %float 6",
            "%f7 = OpConstant %float 7",
            "%col0 = OpConstantComposite %vec2 %f0 %f1",
            "%col1 = OpConstantComposite %vec2 %f2 %f3",
            "%col2 = OpConstantComposite %vec2 %f4 %f5",
            "%col3 = OpConstantComposite %vec2 %f6 %f7",
            "%mat_a = OpConstantComposite %mat2 %col0 %col1",
            "%mat_b = OpConstantComposite %mat2 %col2 %col3",
            "%arr_val = OpConstantComposite %arr %mat_a %mat_b",
            "%value = OpConstantComposite %struct %arr_val",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%elem = OpCompositeExtract %float %value 0 1 0 1",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        assemble_instructions(&refs).expect("assemble instructions");
    }

    #[test]
    fn composite_extract_reports_out_of_range_index() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v = OpCompositeConstruct %vec2 %zero %one",
            "%elem = OpCompositeExtract %int %v 3",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "Composite extract index 3 exceeds vector width 2"
        );
    }

    #[test]
    fn composite_extract_rejects_array_index_out_of_bounds() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%four = OpConstant %int 4",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%two = OpConstant %int 2",
            "%three = OpConstant %int 3",
            "%arr = OpTypeArray %int %four",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%value = OpCompositeConstruct %arr %zero %one %two %three",
            "%elem = OpCompositeExtract %int %value 5",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "Array index 5 exceeds array length 4"
        );
    }

    #[test]
    fn composite_extract_rejects_matrix_column_index() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%float = OpTypeFloat 32",
            "%vec2 = OpTypeVector %float 2",
            "%mat2 = OpTypeMatrix %vec2 2",
            "%f0 = OpConstant %float 0",
            "%f1 = OpConstant %float 1",
            "%col0 = OpConstantComposite %vec2 %f0 %f1",
            "%col1 = OpConstantComposite %vec2 %f1 %f0",
            "%mat = OpConstantComposite %mat2 %col0 %col1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%elem = OpCompositeExtract %vec2 %mat 3",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "Matrix column index 3 exceeds column count 2"
        );
    }

    #[test]
    fn composite_insert_requires_matching_object_type() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%v = OpCompositeConstruct %vec2 %zero %one",
            "%result = OpCompositeInsert %vec2 %v %v 0",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let diagnostics = assemble_instructions(&refs)
            .expect_err("expected diagnostics")
            .into_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "Object operand type must match the selected component type"
        );
    }

    #[test]
    fn translator_emits_constant_composite_and_null() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%vec2 = OpTypeVector %int 2",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%vec_const = OpConstantComposite %vec2 %zero %one",
            "%null_vec = OpConstantNull %vec2",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let mut const_opcodes = module.types_global_values.iter().filter(|inst| {
            inst.class.opcode == spirv::Op::ConstantComposite
                || inst.class.opcode == spirv::Op::ConstantNull
        });
        assert!(const_opcodes.any(|inst| inst.class.opcode == spirv::Op::ConstantComposite));
        assert!(const_opcodes.any(|inst| inst.class.opcode == spirv::Op::ConstantNull));
    }

    #[test]
    fn translator_emits_phi_instruction() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%true = OpConstantTrue %bool",
            "%zero = OpConstant %int 0",
            "%one = OpConstant %int 1",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "OpBranchConditional %true %then %else",
            "%then = OpLabel",
            "OpBranch %merge",
            "%else = OpLabel",
            "OpBranch %merge",
            "%merge = OpLabel",
            "%phi = OpPhi %int %zero %then %one %else",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let phi = function
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .find(|inst| inst.class.opcode == spirv::Op::Phi)
            .expect("phi instruction");
        assert_eq!(phi.operands.len(), 4);
    }

    #[test]
    fn assembler_stamps_target_env_version() {
        let binary =
            assemble_text_with_env("", TargetEnv::Universal1_0).expect("assemble text with env");
        assert!(binary.len() > 1);
        assert_eq!(binary[1], SpirvVersion::new(1, 0).to_word());
    }

    #[test]
    fn assemble_with_spans_tracks_result_ids() {
        use super::assemble_text_with_spans;

        let text = r#"OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn_type = OpTypeFunction %void
%main = OpFunction %void None %fn_type
%entry = OpLabel
OpReturn
OpFunctionEnd"#;

        let result = assemble_text_with_spans(text).expect("assembly should succeed");

        // Verify the span map has entries for the result IDs
        assert!(!result.span_map.is_empty(), "span map should not be empty");

        // Check that we can look up spans for specific IDs
        // %void should be ID 1, %fn_type should be ID 2, etc.
        // The exact IDs depend on resolution order but we should have multiple entries
        assert!(
            result.span_map.id_count() >= 4,
            "should have at least 4 ID spans (void, fn_type, main, entry)"
        );
    }

    #[test]
    fn assemble_with_spans_records_correct_line_info() {
        use super::assemble_text_with_spans;
        use crate::validation::span::SourceLocation;

        let text = "%uint = OpTypeInt 32 0";

        let result = assemble_text_with_spans(text).expect("assembly should succeed");

        // The ID %uint should be resolved to 1
        let span = result
            .span_map
            .get_id_span(1)
            .expect("should have span for ID 1");

        // The span should point to line 0 (zero-based), where %uint is defined
        match span.start {
            SourceLocation::Text(pos) => {
                assert_eq!(pos.line(), 0, "should be on line 0");
                // Column should point to the start of %uint
                assert_eq!(pos.column(), 0, "should start at column 0");
            }
            _ => panic!("expected text source location"),
        }
    }

    #[test]
    fn translator_emits_bitcast() {
        let source = [
            "OpCapability Shader",
            "OpCapability Int8",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "OpExecutionMode %main OriginUpperLeft",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%u8 = OpTypeInt 8 0",
            "%u32 = OpTypeInt 32 0",
            "%c32 = OpConstant %u32 255",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%result = OpBitcast %u8 %c32",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let bitcast = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Bitcast)
            .expect("bitcast instruction");
        assert!(bitcast.result_type.is_some());
        assert!(bitcast.result_id.is_some());
        assert_eq!(bitcast.operands.len(), 1);
    }

    #[test]
    fn translator_emits_convert_s_to_f() {
        let source = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Fragment %main \"main\"",
            "OpExecutionMode %main OriginUpperLeft",
            "%void = OpTypeVoid",
            "%void_fn = OpTypeFunction %void",
            "%int = OpTypeInt 32 1",
            "%float = OpTypeFloat 32",
            "%ci = OpConstant %int 42",
            "%main = OpFunction %void None %void_fn",
            "%entry = OpLabel",
            "%result = OpConvertSToF %float %ci",
            "OpReturn",
            "OpFunctionEnd",
        ];
        let parsed: Vec<_> = source
            .into_iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble instructions");
        let function = module.functions.first().expect("function");
        let block = function.blocks.first().expect("entry block");
        let convert = block
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::ConvertSToF)
            .expect("convert instruction");
        assert!(convert.result_type.is_some());
        assert!(convert.result_id.is_some());
        assert_eq!(convert.operands.len(), 1);
    }

    // ---------------------------------------------------------------
    // Context-dependent number literal tests (OpConstant / OpSpecConstant)
    // Matching C++ spirv-as: integer text for float types is parsed as
    // the float value, not raw bits.
    // ---------------------------------------------------------------

    /// Helper: assemble text, find the first OpConstant, return its operand.
    fn assemble_and_get_constant_operand(lines: &[&str]) -> dr::Operand {
        let parsed: Vec<_> = lines
            .iter()
            .map(|line| parse_instruction(line).expect("parse"))
            .collect();
        let refs: Vec<_> = parsed.iter().collect();
        let module = assemble_instructions(&refs).expect("assemble");
        module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Constant)
            .expect("OpConstant not found")
            .operands
            .first()
            .expect("OpConstant missing operand")
            .clone()
    }

    #[test]
    fn constant_integer_literal_for_float32_encodes_as_float_value() {
        // "42" with float32 type should encode as float 42.0, not raw bits 0x2A.
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 42",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(42.0_f32.to_bits()));
    }

    #[test]
    fn constant_zero_for_float32_encodes_correctly() {
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 0",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(0.0_f32.to_bits()));
    }

    #[test]
    fn constant_one_for_float32_encodes_correctly() {
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 1",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(1.0_f32.to_bits()));
    }

    #[test]
    fn constant_negative_integer_for_float32_encodes_as_negative_float() {
        // "-1" with float32 type should encode as -1.0f (0xBF800000).
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float -1",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32((-1.0_f32).to_bits()));
    }

    #[test]
    fn constant_large_integer_for_float32_encodes_as_float() {
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 1000000",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(1_000_000.0_f32.to_bits()));
    }

    #[test]
    fn constant_integer_literal_for_float64_encodes_as_double_value() {
        // "42" with float64 type should encode as double 42.0.
        let operand = assemble_and_get_constant_operand(&[
            "%double = OpTypeFloat 64",
            "%c = OpConstant %double 42",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit64(42.0_f64.to_bits()));
    }

    #[test]
    fn constant_negative_integer_for_float64_encodes_as_negative_double() {
        let operand = assemble_and_get_constant_operand(&[
            "%double = OpTypeFloat 64",
            "%c = OpConstant %double -1",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit64((-1.0_f64).to_bits()));
    }

    #[test]
    fn constant_float_text_for_float32_encodes_correctly() {
        // "42.5" is float text, parsed via OperandValue::Word path.
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 42.5",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(42.5_f32.to_bits()));
    }

    #[test]
    fn constant_negative_float_text_for_float32_encodes_correctly() {
        let operand = assemble_and_get_constant_operand(&[
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float -3.14",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32((-3.14_f32).to_bits()));
    }

    #[test]
    fn constant_float_text_for_float64_encodes_correctly() {
        let operand = assemble_and_get_constant_operand(&[
            "%double = OpTypeFloat 64",
            "%c = OpConstant %double 42.5",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit64(42.5_f64.to_bits()));
    }

    #[test]
    fn constant_integer_for_uint32_encodes_as_raw_bits() {
        // Integer types should still encode as raw integer bits.
        let operand = assemble_and_get_constant_operand(&[
            "%uint = OpTypeInt 32 0",
            "%c = OpConstant %uint 42",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(42));
    }

    #[test]
    fn constant_integer_for_sint32_encodes_as_raw_bits() {
        let operand = assemble_and_get_constant_operand(&[
            "%int = OpTypeInt 32 1",
            "%c = OpConstant %int 42",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(42));
    }

    #[test]
    fn constant_negative_for_sint32_encodes_twos_complement() {
        let operand = assemble_and_get_constant_operand(&[
            "%int = OpTypeInt 32 1",
            "%c = OpConstant %int -1",
        ]);
        // -1 as i32 in two's complement is 0xFFFFFFFF
        assert_eq!(operand, dr::Operand::LiteralBit32((-1_i32) as u32));
    }

    #[test]
    fn constant_integer_for_uint64_encodes_value() {
        // Small values that fit in 32 bits are stored as LiteralBit32 by
        // encode_literal_operand. This is a pre-existing behavior; the binary
        // serializer handles type-width encoding.
        let operand = assemble_and_get_constant_operand(&[
            "%ulong = OpTypeInt 64 0",
            "%c = OpConstant %ulong 42",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit32(42));
    }

    #[test]
    fn constant_large_integer_for_uint64_encodes_as_64bit() {
        // Values that don't fit in 32 bits should use LiteralBit64.
        let operand = assemble_and_get_constant_operand(&[
            "%ulong = OpTypeInt 64 0",
            "%c = OpConstant %ulong 4294967296",
        ]);
        assert_eq!(operand, dr::Operand::LiteralBit64(4_294_967_296));
    }

    #[test]
    fn constant_float_round_trips_through_assemble_disassemble() {
        // Full round-trip: assemble "OpConstant %float 42" then disassemble
        // and verify the output shows 42 (the float value), not 5.88545e-44.
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 42",
        ]
        .join("\n");
        let disassembled = round_trip_with_options(
            &text,
            TextToBinaryOptions::NONE,
            BinaryToTextOptions::NO_HEADER,
        );
        assert!(
            disassembled.contains("OpConstant") && disassembled.contains(" 42"),
            "Expected disassembly to contain 'OpConstant ... 42', got: {disassembled}"
        );
        // Must NOT contain the subnormal float that 0x2A bit pattern represents
        assert!(
            !disassembled.contains("5.88545"),
            "OpConstant should not show raw bits interpretation: {disassembled}"
        );
    }

    #[test]
    fn constant_float_text_round_trips_through_assemble_disassemble() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%float = OpTypeFloat 32",
            "%c = OpConstant %float 42.5",
        ]
        .join("\n");
        let disassembled = round_trip_with_options(
            &text,
            TextToBinaryOptions::NONE,
            BinaryToTextOptions::NO_HEADER,
        );
        assert!(
            disassembled.contains("42.5"),
            "Expected disassembly to contain '42.5', got: {disassembled}"
        );
    }

    #[test]
    fn constant_does_not_note_integer_constant_for_float_type() {
        // When the type is float, note_integer_constant should NOT be called.
        // Verify by checking that a subsequent array-length lookup doesn't
        // confuse float bits with integer values.
        let type_inst = parse_instruction("%float = OpTypeFloat 32").unwrap();
        let const_inst = parse_instruction("%c = OpConstant %float 42").unwrap();
        let mut translator = AssemblyTranslator::new();
        translator.translate(&type_inst);
        translator.translate(&const_inst);
        let (module, diagnostics) = translator.finish();
        assert!(diagnostics.is_empty());
        let constant = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == spirv::Op::Constant)
            .expect("constant");
        // The operand should be 42.0f bits, not integer 42
        assert_eq!(
            constant.operands.first().unwrap(),
            &dr::Operand::LiteralBit32(42.0_f32.to_bits())
        );
    }
}
