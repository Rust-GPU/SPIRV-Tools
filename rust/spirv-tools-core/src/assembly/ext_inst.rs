use rspirv::grammar::{
    ExtendedInstruction, GlslStd450InstructionTable, LogicalOperand, OpenCLStd100InstructionTable,
};

/// Known extended instruction sets that the assembler understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtInstSetKind {
    /// The `GLSL.std.450` canonical extended instruction set.
    GlslStd450,
    /// The `OpenCL.std` (100) extended instruction set.
    OpenClStd100,
    /// An instruction set without embedded grammar metadata.
    Unknown,
}

impl ExtInstSetKind {
    /// Returns the canonical kind for the provided import name.
    pub fn from_import(name: &str) -> Self {
        match name {
            "GLSL.std.450" => Self::GlslStd450,
            "OpenCL.std" | "OpenCL.std.100" => Self::OpenClStd100,
            _ => Self::Unknown,
        }
    }

    /// Returns the grammar entry for the given opcode name if this set knows about it.
    pub fn lookup(&self, opname: &str) -> Option<&'static ExtendedInstruction<'static>> {
        match self {
            ExtInstSetKind::GlslStd450 => {
                GlslStd450InstructionTable::iter().find(|inst| inst.opname == opname)
            }
            ExtInstSetKind::OpenClStd100 => {
                OpenCLStd100InstructionTable::iter().find(|inst| inst.opname == opname)
            }
            ExtInstSetKind::Unknown => None,
        }
    }

    /// Looks up an extended instruction by numeric opcode.
    pub fn lookup_by_opcode(&self, opcode: u32) -> Option<&'static ExtendedInstruction<'static>> {
        match self {
            ExtInstSetKind::GlslStd450 => {
                GlslStd450InstructionTable::iter().find(|inst| inst.opcode == opcode)
            }
            ExtInstSetKind::OpenClStd100 => {
                OpenCLStd100InstructionTable::iter().find(|inst| inst.opcode == opcode)
            }
            ExtInstSetKind::Unknown => None,
        }
    }

    /// Returns true if this instruction set ships with embedded grammar metadata.
    pub fn has_grammar(self) -> bool {
        !matches!(self, ExtInstSetKind::Unknown)
    }
}

/// Information tracked for each imported extended instruction set.
#[derive(Debug, Clone)]
pub struct ExtInstImportInfo {
    /// Classified instruction set kind.
    pub kind: ExtInstSetKind,
    /// Canonical name provided to `OpExtInstImport`.
    pub name: String,
}

impl ExtInstImportInfo {
    /// Creates import metadata for the provided instruction set name.
    pub fn new(name: &str) -> Self {
        Self {
            kind: ExtInstSetKind::from_import(name),
            name: name.to_string(),
        }
    }
}

/// Result of resolving an `OpExtInst` opcode.
#[derive(Clone, Copy)]
pub struct ResolvedExtInst<'a> {
    /// Numeric opcode for the resolved instruction.
    pub opcode: u32,
    /// Optional operand descriptors describing the extended instruction shape.
    pub operands: Option<&'a [LogicalOperand]>,
}

impl<'a> From<&'a ExtendedInstruction<'static>> for ResolvedExtInst<'a> {
    fn from(inst: &'a ExtendedInstruction<'static>) -> Self {
        Self {
            opcode: inst.opcode,
            operands: Some(inst.operands),
        }
    }
}
