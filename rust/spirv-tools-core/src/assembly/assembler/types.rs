use thiserror::Error;

use crate::diagnostic::MessagePosition;

#[derive(Debug, Error)]
pub(super) enum MemberDecorationError {
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
    pub(super) field_types: Vec<u32>,
    pub(super) member_layouts: Vec<MemberLayout>,
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
pub(super) enum MatrixMajorness {
    RowMajor,
    ColumnMajor,
}

impl MatrixMajorness {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            MatrixMajorness::RowMajor => "RowMajor",
            MatrixMajorness::ColumnMajor => "ColMajor",
        }
    }
}

/// Captures matrix layout metadata attached to a struct member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemberLayout {
    pub(super) majorness: Option<MemberMajorness>,
    pub(super) matrix_stride: Option<MemberMatrixStride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemberMajorness {
    pub(super) kind: MatrixMajorness,
    pub(super) position: MessagePosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemberMatrixStride {
    pub(super) value: u32,
    pub(super) position: MessagePosition,
}
