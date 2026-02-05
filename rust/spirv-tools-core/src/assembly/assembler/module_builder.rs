use core::convert::TryFrom;
use std::borrow::Cow;
use std::collections::BTreeMap;

use super::types::{
    ArrayTypeInfo, CompositeTypeInfo, MatrixMajorness, MatrixTypeInfo, MemberDecorationError,
    MemberMajorness, MemberMatrixStride, StructTypeInfo, VectorTypeInfo,
};
use crate::assembly::ext_inst::ExtInstImportInfo;
use crate::assembly::instruction::{IdRef, ResultId, SpirvId, TypeId};
use crate::diagnostic::{DiagnosticMessage, MessagePosition};
use crate::message::MessageLevel;
use crate::validation::span::{SourceSpan, SpanMap};

/// Tracks textual identifiers and diagnostics while constructing a module.
#[derive(Debug)]
pub struct ModuleBuilder<'a> {
    pub(super) named_ids: BTreeMap<&'a str, u32>,
    pub(super) numeric_ids: BTreeMap<u32, u32>,
    pub(super) next_numeric_id: u32,
    pub(super) diagnostics: Vec<DiagnosticMessage<'static>>,
    pub(super) value_types: BTreeMap<u32, u32>,
    pub(super) composite_types: BTreeMap<u32, CompositeTypeInfo>,
    pub(super) integer_constants: BTreeMap<u32, u64>,
    pub(super) ext_inst_imports: BTreeMap<u32, ExtInstImportInfo>,
    pub(super) preserve_numeric_ids: bool,
    /// Optional span map for tracking source locations of IDs.
    pub(super) span_map: Option<SpanMap>,
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

    pub(super) fn resolve_spirv_id(&mut self, id: SpirvId<'a>) -> u32 {
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

    #[allow(dead_code)]
    pub(super) fn bind_result_id(&mut self, result_id: ResultId<'a>, numeric: u32) {
        self.bind_spirv_id(result_id.as_spirv_id(), numeric);
    }

    #[allow(dead_code)]
    pub(super) fn bind_spirv_id(&mut self, id: SpirvId<'a>, numeric: u32) {
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

    pub(super) fn bind_typed_result(
        &mut self,
        result_type: TypeId<'a>,
        result_id: ResultId<'a>,
    ) -> (u32, u32) {
        let type_id = self.resolve_type_id(result_type);
        let value_id = self.resolve_result_id(result_id);
        self.value_types.insert(value_id, type_id);
        (type_id, value_id)
    }

    #[allow(dead_code)]
    pub(super) fn note_numeric_result_type(&mut self, value_id: u32, type_id: u32) {
        self.value_types.insert(value_id, type_id);
    }

    pub(super) fn value_type(&self, value_id: u32) -> Option<u32> {
        self.value_types.get(&value_id).copied()
    }

    pub(super) fn composite_type(&self, type_id: u32) -> Option<&CompositeTypeInfo> {
        self.composite_types.get(&type_id)
    }

    pub(super) fn vector_type(&self, type_id: u32) -> Option<VectorTypeInfo> {
        self.composite_types
            .get(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Vector(vector) => Some(*vector),
                _ => None,
            })
    }

    pub(super) fn note_vector_type(&mut self, type_id: u32, info: VectorTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Vector(info));
    }

    pub(super) fn note_array_type(&mut self, type_id: u32, info: ArrayTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Array(info));
    }

    pub(super) fn note_struct_type(&mut self, type_id: u32, info: StructTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Struct(info));
    }

    pub(super) fn note_matrix_type(&mut self, type_id: u32, info: MatrixTypeInfo) {
        self.composite_types
            .insert(type_id, CompositeTypeInfo::Matrix(info));
    }

    pub(super) fn array_length(&self, info: &ArrayTypeInfo) -> Option<u32> {
        self.integer_constants
            .get(&info.length_constant())
            .and_then(|value| u32::try_from(*value).ok())
    }

    pub(super) fn note_integer_constant(&mut self, result_id: u32, value: u64) {
        self.integer_constants.insert(result_id, value);
    }

    pub(super) fn struct_info(&self, type_id: u32) -> Option<&StructTypeInfo> {
        self.composite_types
            .get(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Struct(struct_info) => Some(struct_info),
                _ => None,
            })
    }

    pub(super) fn struct_info_mut(&mut self, type_id: u32) -> Option<&mut StructTypeInfo> {
        self.composite_types
            .get_mut(&type_id)
            .and_then(|info| match info {
                CompositeTypeInfo::Struct(struct_info) => Some(struct_info),
                _ => None,
            })
    }

    pub(super) fn resolve_struct_member_type(
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

    pub(super) fn type_contains_matrix(&self, type_id: u32) -> bool {
        match self.composite_types.get(&type_id) {
            Some(CompositeTypeInfo::Matrix(_)) => true,
            Some(CompositeTypeInfo::Array(info)) => self.type_contains_matrix(info.element_type()),
            _ => false,
        }
    }

    pub(super) fn apply_member_majorness(
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

    pub(super) fn apply_member_matrix_stride(
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
