use core::num::NonZeroU32;
use rspirv::grammar::{CoreInstructionTable, Instruction as GrammarInstruction, LogicalOperand};
use rspirv::grammar::{OperandKind, OperandQuantifier};
use rspirv::spirv;

use super::lexer::{NamedId, Span};
#[cfg(test)]
use crate::diagnostic::MessagePosition;
/// Wrapper around the SPIR-V grammar for a single instruction.
#[derive(Clone, Copy, Debug)]
pub struct InstructionLayout {
    grammar: &'static GrammarInstruction<'static>,
    payload_start: usize,
    result_type: Option<OperandDescriptor>,
    result_id: Option<OperandDescriptor>,
}

impl InstructionLayout {
    /// Creates a layout description for the given opcode if it exists in the core grammar.
    pub fn lookup(opcode: spirv::Op) -> Option<Self> {
        let grammar = CoreInstructionTable::iter().find(|inst| inst.opcode == opcode)?;
        let (result_type, result_id, payload_start) = partition_operands(grammar.operands);
        Some(Self {
            grammar,
            payload_start,
            result_type,
            result_id,
        })
    }

    /// Returns the opcode backing this layout.
    pub fn opcode(&self) -> spirv::Op {
        self.grammar.opcode
    }

    /// Returns the operand descriptor for the result type, if any.
    pub fn result_type(&self) -> Option<OperandDescriptor> {
        self.result_type
    }

    /// Returns the operand descriptor for the result id, if any.
    pub fn result_id(&self) -> Option<OperandDescriptor> {
        self.result_id
    }

    /// Returns an iterator over the payload operand descriptors (excluding result type/id).
    pub fn operands(&self) -> OperandIter<'_> {
        OperandIter {
            inner: &self.grammar.operands[self.payload_start..],
            index: 0,
        }
    }

    /// Returns the logical operands slice backing this layout.
    pub fn operand_slice(&self) -> &'static [LogicalOperand] {
        self.grammar.operands
    }
}

/// Typed descriptor mirroring the SPIR-V grammar operand metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandDescriptor {
    kind: OperandKind,
    quantifier: OperandQuantifier,
}

impl OperandDescriptor {
    /// Creates a descriptor from the grammar representation.
    pub(crate) const fn new(kind: OperandKind, quantifier: OperandQuantifier) -> Self {
        Self { kind, quantifier }
    }

    /// Returns the operand kind defined by the grammar.
    pub const fn kind(&self) -> OperandKind {
        self.kind
    }

    /// Returns the quantifier describing how many times this operand can repeat.
    pub const fn quantifier(&self) -> OperandQuantifier {
        self.quantifier
    }

    /// Returns true if this operand is optional.
    pub const fn is_optional(&self) -> bool {
        matches!(
            self.quantifier,
            OperandQuantifier::ZeroOrOne | OperandQuantifier::ZeroOrMore
        )
    }
}

/// Iterator over payload operand descriptors for a given instruction layout.
pub struct OperandIter<'a> {
    inner: &'a [LogicalOperand],
    index: usize,
}

impl<'a> Iterator for OperandIter<'a> {
    type Item = OperandDescriptor;

    fn next(&mut self) -> Option<Self::Item> {
        let operand = self.inner.get(self.index)?;
        self.index += 1;
        Some(OperandDescriptor::from_logical(operand))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.inner.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OperandIter<'_> {}

impl OperandDescriptor {
    fn from_logical(operand: &LogicalOperand) -> Self {
        Self::new(operand.kind, operand.quantifier)
    }
}

fn partition_operands(
    operands: &'static [LogicalOperand],
) -> (Option<OperandDescriptor>, Option<OperandDescriptor>, usize) {
    let mut index = 0;
    let mut result_type = None;
    let mut result_id = None;

    if let Some(op) = operands.get(index) {
        if op.kind == OperandKind::IdResultType {
            result_type = Some(OperandDescriptor::from_logical(op));
            index += 1;
        }
    }

    if let Some(op) = operands.get(index) {
        if op.kind == OperandKind::IdResult {
            result_id = Some(OperandDescriptor::from_logical(op));
            index += 1;
        }
    }

    (result_type, result_id, index)
}

/// Textual or numeric SPIR-V identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpirvId<'a> {
    /// References a textual named identifier (e.g. `%value`).
    Named(NamedId<'a>),
    /// References an already-materialized numeric identifier.
    Numeric(NonZeroU32),
}

impl<'a> SpirvId<'a> {
    /// Wraps a named identifier.
    pub const fn named(id: NamedId<'a>) -> Self {
        Self::Named(id)
    }

    /// Wraps a numeric identifier.
    pub const fn numeric(id: NonZeroU32) -> Self {
        Self::Numeric(id)
    }

    /// Returns the named identifier if present.
    pub fn as_named(&self) -> Option<NamedId<'a>> {
        match self {
            SpirvId::Named(id) => Some(*id),
            SpirvId::Numeric(_) => None,
        }
    }

    /// Returns the numeric identifier if present.
    pub fn as_numeric(&self) -> Option<NonZeroU32> {
        match self {
            SpirvId::Named(_) => None,
            SpirvId::Numeric(id) => Some(*id),
        }
    }
}

macro_rules! id_newtype {
    ($name:ident) => {
        #[doc = concat!("Typed SPIR-V ", stringify!($name), " identifier.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name<'a> {
            id: SpirvId<'a>,
            span: Span,
        }

        impl<'a> $name<'a> {
            /// Wraps the provided identifier and its source span.
            pub const fn new(id: SpirvId<'a>, span: Span) -> Self {
                Self { id, span }
            }

            /// Returns the underlying representation.
            pub const fn as_spirv_id(self) -> SpirvId<'a> {
                self.id
            }

            /// Returns the span covering this identifier.
            pub const fn span(self) -> Span {
                self.span
            }
        }
    };
}

id_newtype!(ResultId);
id_newtype!(TypeId);
id_newtype!(IdRef);

/// Numeric literal captured during assembly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LiteralNumber {
    /// Unsigned integer literal.
    Unsigned(u64),
    /// Signed integer literal.
    Signed(i64),
}

impl LiteralNumber {
    /// Creates an unsigned literal from the given value.
    pub const fn unsigned(value: u64) -> Self {
        Self::Unsigned(value)
    }

    /// Creates a signed literal from the given value.
    pub const fn signed(value: i64) -> Self {
        Self::Signed(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_instruction_layout_has_result_type_and_id() {
        let layout = InstructionLayout::lookup(spirv::Op::Load).expect("layout");
        assert_eq!(
            layout.result_type().unwrap().kind(),
            OperandKind::IdResultType
        );
        assert_eq!(layout.result_id().unwrap().kind(), OperandKind::IdResult);
        let kinds: Vec<_> = layout.operands().map(|operand| operand.kind()).collect();
        assert_eq!(kinds, vec![OperandKind::IdRef, OperandKind::MemoryAccess]);
    }

    #[test]
    fn store_instruction_layout_has_no_results() {
        let layout = InstructionLayout::lookup(spirv::Op::Store).expect("layout");
        assert!(layout.result_type().is_none());
        assert!(layout.result_id().is_none());
        let kinds: Vec<_> = layout.operands().map(|operand| operand.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                OperandKind::IdRef,
                OperandKind::IdRef,
                OperandKind::MemoryAccess
            ]
        );
    }

    #[test]
    fn type_int_layout_reports_result_id_only() {
        let layout = InstructionLayout::lookup(spirv::Op::TypeInt).expect("layout");
        assert!(layout.result_type().is_none());
        assert_eq!(layout.result_id().unwrap().kind(), OperandKind::IdResult);
        let mut iter = layout.operands();
        assert_eq!(iter.next().unwrap().kind(), OperandKind::LiteralInteger);
        assert_eq!(iter.next().unwrap().kind(), OperandKind::LiteralInteger);
        assert!(iter.next().is_none());
    }

    #[test]
    fn id_wrappers_store_numeric_ids() {
        let numeric = NonZeroU32::new(42).unwrap();
        let span = Span::from_point(MessagePosition::new(0, 0, 0));
        let result_id = ResultId::new(SpirvId::numeric(numeric), span);
        assert_eq!(result_id.as_spirv_id().as_numeric(), Some(numeric));
    }

    #[test]
    fn literal_number_constructors_work() {
        assert_eq!(LiteralNumber::unsigned(5), LiteralNumber::Unsigned(5));
        assert_eq!(LiteralNumber::signed(-7), LiteralNumber::Signed(-7));
    }
}
