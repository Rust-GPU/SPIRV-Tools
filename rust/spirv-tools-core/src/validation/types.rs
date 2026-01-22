//! SPIR-V ID types and wrapper newtypes.
//!
//! This module provides strongly-typed wrappers around SPIR-V IDs to prevent
//! mixing different ID categories and to make illegal states unrepresentable.

use std::{fmt, num::NonZeroU32, sync::Arc};

/// Errors produced when attempting to construct zero-valued ids.
#[derive(Debug, thiserror::Error, Copy, Clone, PartialEq, Eq)]
#[error("ids must be non-zero")]
pub struct ZeroIdError;

/// A non-zero SPIR-V id.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU32);

impl Id {
    /// Wraps an existing non-zero id.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Returns the underlying non-zero id.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<Id> for u32 {
    fn from(id: Id) -> Self {
        id.0.get()
    }
}

impl From<NonZeroU32> for Id {
    fn from(value: NonZeroU32) -> Self {
        Id::new(value)
    }
}

impl TryFrom<u32> for Id {
    type Error = ZeroIdError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        NonZeroU32::new(value).map(Id).ok_or(ZeroIdError)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Result ids must be non-zero and unique within a module.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ResultId(Id);

/// Type ids referenced by instructions (non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(Id);

/// Operand ids appearing in instruction operands (non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OperandId(Id);

macro_rules! id_wrapper {
    ($name:ident) => {
        impl $name {
            /// Wraps a non-zero `Id` in the typed wrapper.
            pub fn new(id: Id) -> Self {
                Self(id)
            }

            /// Unwraps the inner `Id`.
            pub fn into_inner(self) -> Id {
                self.0
            }
        }

        impl TryFrom<u32> for $name {
            type Error = ZeroIdError;

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                Id::try_from(value).map(Self)
            }
        }

        impl From<NonZeroU32> for $name {
            fn from(value: NonZeroU32) -> Self {
                Self(Id::new(value))
            }
        }

        impl From<$name> for Id {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.0.get()
            }
        }
    };
}

id_wrapper!(ResultId);
id_wrapper!(TypeId);
id_wrapper!(OperandId);

/// Decoration targets (non-zero ids referenced by decoration instructions).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DecorationTargetId(OperandId);

impl DecorationTargetId {
    /// Wraps a non-zero operand id in a decoration target.
    pub fn new(id: OperandId) -> Self {
        Self(id)
    }

    /// Returns the underlying operand id.
    pub fn into_inner(self) -> OperandId {
        self.0
    }
}

impl From<NonZeroU32> for DecorationTargetId {
    fn from(value: NonZeroU32) -> Self {
        DecorationTargetId::new(OperandId::from(value))
    }
}

impl From<DecorationTargetId> for Id {
    fn from(value: DecorationTargetId) -> Self {
        value.0.into_inner()
    }
}

impl TryFrom<u32> for DecorationTargetId {
    type Error = ZeroIdError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        OperandId::try_from(value).map(DecorationTargetId::new)
    }
}

impl fmt::Display for DecorationTargetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id: Id = (*self).into();
        id.fmt(f)
    }
}

/// A struct member index (can be zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemberIndex(pub u32);

impl MemberIndex {
    /// Constructs a member index from a raw literal (zero is allowed).
    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the underlying literal member index.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for MemberIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Member decoration targets capture the struct id plus the member index.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MemberDecorationTargetId {
    target: DecorationTargetId,
    member: MemberIndex,
}

impl MemberDecorationTargetId {
    /// Creates a member decoration target from a struct id and member index.
    pub fn new(target: DecorationTargetId, member: MemberIndex) -> Self {
        Self { target, member }
    }

    /// Returns the struct id being decorated.
    pub fn target(self) -> DecorationTargetId {
        self.target
    }

    /// Returns the member index being decorated.
    pub fn member(self) -> MemberIndex {
        self.member
    }
}

/// The schema (reserved word) from the module header.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Schema(u32);

impl Schema {
    /// The only valid schema value; SPIR-V reserves this header field.
    pub const ZERO: Schema = Schema(0);

    /// Validates the raw schema value from the module header.
    pub fn validate(raw: u32) -> Result<Self, super::ValidationError> {
        if raw == 0 {
            Ok(Schema::ZERO)
        } else {
            Err(super::ValidationError::InvalidReservedWord { reserved: raw })
        }
    }

    /// Returns the raw schema value (always zero for valid modules).
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A declared upper bound for SPIR-V ids (must be non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IdBound(NonZeroU32);

impl IdBound {
    /// Wraps an existing non-zero bound.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Attempts to create an id bound from a raw value, returning `None` if zero.
    pub fn from_raw(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    /// Returns the underlying non-zero bound value.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<IdBound> for u32 {
    fn from(bound: IdBound) -> Self {
        bound.0.get()
    }
}

impl From<NonZeroU32> for IdBound {
    fn from(value: NonZeroU32) -> Self {
        IdBound::new(value)
    }
}

impl fmt::Display for IdBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A declared (possibly zero) id bound from a module header.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DeclaredBound(pub u32);

impl std::fmt::Display for DeclaredBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<u32> for DeclaredBound {
    fn from(value: u32) -> Self {
        DeclaredBound(value)
    }
}

/// A validated (non-zero) id bound paired with the originally declared value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CheckedBound {
    declared: DeclaredBound,
    validated: IdBound,
}

impl CheckedBound {
    /// Creates a checked bound from a declared bound, returning `None` when the declared value is zero.
    pub fn new(declared: DeclaredBound) -> Option<Self> {
        IdBound::from_raw(declared.0).map(|validated| Self {
            declared,
            validated,
        })
    }

    /// Returns the originally declared bound (which may be zero).
    pub fn declared(self) -> DeclaredBound {
        self.declared
    }

    /// Returns the validated, non-zero bound.
    pub fn validated(self) -> IdBound {
        self.validated
    }
}

impl fmt::Display for CheckedBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.declared.fmt(f)
    }
}

/// Shared, validated words backing a module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleWords(Arc<[u32]>);

impl ModuleWords {
    /// Wraps already-owned SPIR-V words.
    pub fn new(words: Arc<[u32]>) -> Self {
        Self(words)
    }

    /// Clones the shared words as a slice reference.
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Returns a shared reference-counted handle to the words.
    pub fn shared(&self) -> Arc<[u32]> {
        Arc::clone(&self.0)
    }

    /// Consumes the wrapper and returns the underlying `Arc`.
    pub fn into_arc(self) -> Arc<[u32]> {
        self.0
    }
}

impl From<Arc<[u32]>> for ModuleWords {
    fn from(words: Arc<[u32]>) -> Self {
        ModuleWords::new(words)
    }
}

impl From<Box<[u32]>> for ModuleWords {
    fn from(words: Box<[u32]>) -> Self {
        ModuleWords::new(words.into())
    }
}

impl From<ModuleWords> for Arc<[u32]> {
    fn from(words: ModuleWords) -> Self {
        words.into_arc()
    }
}

impl AsRef<[u32]> for ModuleWords {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

/// Strongly-typed extension name to avoid raw string misuse.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ExtensionName(String);

impl ExtensionName {
    /// Returns the underlying extension name as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for ExtensionName {
    fn from(value: &str) -> Self {
        ExtensionName(value.to_string())
    }
}

impl From<String> for ExtensionName {
    fn from(value: String) -> Self {
        ExtensionName(value)
    }
}

impl std::fmt::Display for ExtensionName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Categories of ids that must be non-zero.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IdKind {
    /// Result id produced by an instruction.
    Result,
    /// Result type id associated with an instruction.
    ResultType,
    /// Ids that appear within operands.
    Operand,
}

impl fmt::Display for IdKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdKind::Result => write!(f, "result id"),
            IdKind::ResultType => write!(f, "result type id"),
            IdKind::Operand => write!(f, "operand id"),
        }
    }
}

/// Identifies the role of a merge target for diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeTargetKind {
    /// The merge block target.
    Merge,
    /// The loop continue target.
    Continue,
}

impl std::fmt::Display for MergeTargetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeTargetKind::Merge => f.write_str("merge"),
            MergeTargetKind::Continue => f.write_str("continue"),
        }
    }
}

/// Categories of targets required by specific decorations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DecorationTargetKind {
    /// Struct type targets.
    StructType,
    /// Array, runtime array, or pointer types.
    ArrayOrPointerType,
    /// Functions.
    Function,
    /// Functions or variables.
    FunctionOrVariable,
    /// Variable-like declarations (variables and untyped variables).
    Variable,
    /// Memory object declarations (variables, parameters, raw access chains).
    MemoryObjectDeclaration,
    /// Pointer types.
    Pointer,
    /// Scalar specialization constants.
    ScalarSpecConstant,
    /// Non-specialization constants.
    Constant,
}

impl fmt::Display for DecorationTargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorationTargetKind::StructType => write!(f, "struct type"),
            DecorationTargetKind::ArrayOrPointerType => {
                write!(f, "array, runtime array, or pointer type")
            }
            DecorationTargetKind::Function => write!(f, "function"),
            DecorationTargetKind::FunctionOrVariable => write!(f, "function or variable"),
            DecorationTargetKind::Variable => write!(f, "variable"),
            DecorationTargetKind::MemoryObjectDeclaration => {
                write!(f, "memory object declaration")
            }
            DecorationTargetKind::Pointer => write!(f, "pointer type"),
            DecorationTargetKind::ScalarSpecConstant => write!(f, "scalar specialization constant"),
            DecorationTargetKind::Constant => write!(f, "constant"),
        }
    }
}

// ============================================================================
// SPIR-V Type System Types
// ============================================================================

/// Bit width for numeric types.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BitWidth(NonZeroU32);

impl BitWidth {
    /// 8-bit width.
    pub const BITS_8: Self = Self(NonZeroU32::new(8).unwrap());
    /// 16-bit width.
    pub const BITS_16: Self = Self(NonZeroU32::new(16).unwrap());
    /// 32-bit width.
    pub const BITS_32: Self = Self(NonZeroU32::new(32).unwrap());
    /// 64-bit width.
    pub const BITS_64: Self = Self(NonZeroU32::new(64).unwrap());

    /// Creates a new bit width from a raw value.
    pub fn new(width: u32) -> Option<Self> {
        NonZeroU32::new(width).map(Self)
    }

    /// Returns the raw bit width value.
    pub fn get(self) -> u32 {
        self.0.get()
    }

    /// Returns true if this is 32-bit.
    pub fn is_32(self) -> bool {
        self.0.get() == 32
    }
}

impl fmt::Display for BitWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-bit", self.0)
    }
}

/// Vector component count (must be 2, 3, 4, 8, or 16).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VectorSize(u8);

impl VectorSize {
    /// 2-component vector.
    pub const VEC2: Self = Self(2);
    /// 3-component vector.
    pub const VEC3: Self = Self(3);
    /// 4-component vector.
    pub const VEC4: Self = Self(4);
    /// 8-component vector (for extended types).
    pub const VEC8: Self = Self(8);
    /// 16-component vector (for extended types).
    pub const VEC16: Self = Self(16);

    /// Creates a new vector size from a raw value.
    pub fn new(size: u32) -> Option<Self> {
        match size {
            2 | 3 | 4 | 8 | 16 => Some(Self(size as u8)),
            _ => None,
        }
    }

    /// Returns the raw component count.
    pub fn get(self) -> u32 {
        self.0 as u32
    }
}

impl fmt::Display for VectorSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec{}", self.0)
    }
}

/// Matrix column count (must be 2, 3, or 4).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MatrixColumns(u8);

impl MatrixColumns {
    /// 2-column matrix.
    pub const MAT2: Self = Self(2);
    /// 3-column matrix.
    pub const MAT3: Self = Self(3);
    /// 4-column matrix.
    pub const MAT4: Self = Self(4);

    /// Creates a new matrix column count from a raw value.
    pub fn new(cols: u32) -> Option<Self> {
        match cols {
            2..=4 => Some(Self(cols as u8)),
            _ => None,
        }
    }

    /// Returns the raw column count.
    pub fn get(self) -> u32 {
        self.0 as u32
    }
}

impl fmt::Display for MatrixColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mat{}", self.0)
    }
}

/// Scalar type kinds for validation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    /// Boolean type.
    Bool,
    /// Signed integer.
    SignedInt(BitWidth),
    /// Unsigned integer.
    UnsignedInt(BitWidth),
    /// Floating-point.
    Float(BitWidth),
}

impl ScalarKind {
    /// Returns true if this is a boolean.
    pub fn is_bool(self) -> bool {
        matches!(self, ScalarKind::Bool)
    }

    /// Returns true if this is any integer type.
    pub fn is_int(self) -> bool {
        matches!(self, ScalarKind::SignedInt(_) | ScalarKind::UnsignedInt(_))
    }

    /// Returns true if this is an unsigned integer.
    pub fn is_unsigned_int(self) -> bool {
        matches!(self, ScalarKind::UnsignedInt(_))
    }

    /// Returns true if this is a signed integer.
    pub fn is_signed_int(self) -> bool {
        matches!(self, ScalarKind::SignedInt(_))
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, ScalarKind::Float(_))
    }

    /// Returns the bit width for numeric types (None for bool).
    pub fn bit_width(self) -> Option<BitWidth> {
        match self {
            ScalarKind::Bool => None,
            ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w) | ScalarKind::Float(w) => Some(w),
        }
    }
}

impl fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarKind::Bool => write!(f, "bool"),
            ScalarKind::SignedInt(w) => write!(f, "int{}", w.get()),
            ScalarKind::UnsignedInt(w) => write!(f, "uint{}", w.get()),
            ScalarKind::Float(w) => write!(f, "float{}", w.get()),
        }
    }
}

/// Describes the structure of a SPIR-V type for validation purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum TypeStructure {
    /// Void type (for function returns).
    Void,
    /// Scalar type.
    Scalar(ScalarKind),
    /// Vector type with component type and size.
    Vector {
        component: ScalarKind,
        size: VectorSize,
    },
    /// Matrix type with column vector type and column count.
    Matrix {
        component: ScalarKind,
        rows: VectorSize,
        cols: MatrixColumns,
    },
    /// Array type (element type ID and optional length).
    Array {
        element: TypeId,
        length: Option<u64>,
    },
    /// Runtime array (unbounded, element type ID).
    RuntimeArray { element: TypeId },
    /// Struct type (member type IDs).
    Struct { members: Vec<TypeId> },
    /// Pointer type.
    Pointer {
        pointee: Option<TypeId>,
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Image type.
    Image { sampled_type: Option<TypeId> },
    /// Sampler type.
    Sampler,
    /// Sampled image type.
    SampledImage { image_type: TypeId },
    /// Function type.
    Function {
        return_type: TypeId,
        params: Vec<TypeId>,
    },
    /// Cooperative matrix type (KHR or NV).
    CooperativeMatrix { component: TypeId },
    /// Cooperative vector type (NV).
    CooperativeVector { component: TypeId },
    /// Forward pointer declaration.
    ForwardPointer {
        storage_class: rspirv::spirv::StorageClass,
    },
    /// Opaque type.
    Opaque,
    /// Unknown/unsupported type.
    Unknown,
}

impl TypeStructure {
    /// Returns true if this type is numeric (scalar, vector, or matrix with numeric components).
    pub fn is_numeric(&self) -> bool {
        match self {
            TypeStructure::Scalar(k) => !k.is_bool(),
            TypeStructure::Vector { component, .. } => !component.is_bool(),
            TypeStructure::Matrix { .. } => true,
            _ => false,
        }
    }

    /// Returns true if this is a scalar type.
    pub fn is_scalar(&self) -> bool {
        matches!(self, TypeStructure::Scalar(_))
    }

    /// Returns true if this is a vector type.
    pub fn is_vector(&self) -> bool {
        matches!(self, TypeStructure::Vector { .. })
    }

    /// Returns true if this is a matrix type.
    pub fn is_matrix(&self) -> bool {
        matches!(self, TypeStructure::Matrix { .. })
    }

    /// Returns true if this is a bool scalar type.
    pub fn is_bool_scalar(&self) -> bool {
        matches!(self, TypeStructure::Scalar(ScalarKind::Bool))
    }

    /// Returns true if this is a bool scalar or vector.
    pub fn is_bool_scalar_or_vector(&self) -> bool {
        matches!(
            self,
            TypeStructure::Scalar(ScalarKind::Bool)
                | TypeStructure::Vector {
                    component: ScalarKind::Bool,
                    ..
                }
        )
    }

    /// Returns true if this is an int scalar type (signed or unsigned).
    pub fn is_int_scalar(&self) -> bool {
        matches!(
            self,
            TypeStructure::Scalar(ScalarKind::SignedInt(_) | ScalarKind::UnsignedInt(_))
        )
    }

    /// Returns true if this is an int scalar or vector.
    pub fn is_int_scalar_or_vector(&self) -> bool {
        match self {
            TypeStructure::Scalar(k) => k.is_int(),
            TypeStructure::Vector { component, .. } => component.is_int(),
            _ => false,
        }
    }

    /// Returns true if this is an unsigned int scalar type.
    pub fn is_unsigned_int_scalar(&self) -> bool {
        matches!(self, TypeStructure::Scalar(ScalarKind::UnsignedInt(_)))
    }

    /// Returns true if this is an unsigned int scalar or vector.
    pub fn is_unsigned_int_scalar_or_vector(&self) -> bool {
        matches!(
            self,
            TypeStructure::Scalar(ScalarKind::UnsignedInt(_))
                | TypeStructure::Vector {
                    component: ScalarKind::UnsignedInt(_),
                    ..
                }
        )
    }

    /// Returns true if this is a float scalar type.
    pub fn is_float_scalar(&self) -> bool {
        matches!(self, TypeStructure::Scalar(ScalarKind::Float(_)))
    }

    /// Returns true if this is a float scalar or vector.
    pub fn is_float_scalar_or_vector(&self) -> bool {
        matches!(
            self,
            TypeStructure::Scalar(ScalarKind::Float(_))
                | TypeStructure::Vector {
                    component: ScalarKind::Float(_),
                    ..
                }
        )
    }

    /// Returns true if this is a cooperative matrix type.
    pub fn is_cooperative_matrix(&self) -> bool {
        matches!(self, TypeStructure::CooperativeMatrix { .. })
    }

    /// Returns true if this is a cooperative vector type.
    pub fn is_cooperative_vector(&self) -> bool {
        matches!(self, TypeStructure::CooperativeVector { .. })
    }

    /// Returns the scalar kind if this is a scalar, or the component kind if vector.
    pub fn scalar_or_component_kind(&self) -> Option<ScalarKind> {
        match self {
            TypeStructure::Scalar(k) => Some(*k),
            TypeStructure::Vector { component, .. } => Some(*component),
            _ => None,
        }
    }

    /// Returns the bit width if this is a numeric scalar or vector.
    pub fn bit_width(&self) -> Option<BitWidth> {
        self.scalar_or_component_kind().and_then(|k| k.bit_width())
    }

    /// Returns the dimension (1 for scalar, N for vecN).
    pub fn dimension(&self) -> u32 {
        match self {
            TypeStructure::Scalar(_) => 1,
            TypeStructure::Vector { size, .. } => size.get(),
            TypeStructure::Matrix { cols, rows, .. } => cols.get() * rows.get(),
            _ => 1,
        }
    }

    /// Returns the vector size if this is a vector.
    pub fn vector_size(&self) -> Option<VectorSize> {
        match self {
            TypeStructure::Vector { size, .. } => Some(*size),
            _ => None,
        }
    }
}

impl fmt::Display for TypeStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeStructure::Void => write!(f, "void"),
            TypeStructure::Scalar(k) => write!(f, "{}", k),
            TypeStructure::Vector { component, size } => {
                write!(f, "vec{}({})", size.get(), component)
            }
            TypeStructure::Matrix {
                component,
                rows,
                cols,
            } => {
                write!(f, "mat{}x{}({})", cols.get(), rows.get(), component)
            }
            TypeStructure::Array { .. } => write!(f, "array"),
            TypeStructure::RuntimeArray { .. } => write!(f, "runtime_array"),
            TypeStructure::Struct { members } => write!(f, "struct({} members)", members.len()),
            TypeStructure::Pointer { storage_class, .. } => write!(f, "ptr<{:?}>", storage_class),
            TypeStructure::Image { .. } => write!(f, "image"),
            TypeStructure::Sampler => write!(f, "sampler"),
            TypeStructure::SampledImage { .. } => write!(f, "sampled_image"),
            TypeStructure::Function { .. } => write!(f, "function"),
            TypeStructure::CooperativeMatrix { .. } => write!(f, "cooperative_matrix"),
            TypeStructure::CooperativeVector { .. } => write!(f, "cooperative_vector"),
            TypeStructure::ForwardPointer { .. } => write!(f, "forward_pointer"),
            TypeStructure::Opaque => write!(f, "opaque"),
            TypeStructure::Unknown => write!(f, "unknown"),
        }
    }
}
