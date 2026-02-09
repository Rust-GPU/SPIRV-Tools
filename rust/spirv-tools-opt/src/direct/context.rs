//! Context for building egglog expressions from SPIR-V instructions.

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

use super::TypeClass;

/// Context for building egglog expressions from SPIR-V instructions.
pub struct EgglogContext {
    /// Maps result_id -> egglog term string
    pub id_to_term: HashMap<Word, String>,
    /// Maps result_id -> result_type
    pub id_to_type: HashMap<Word, Word>,
    /// Type widths (int, float, and bool)
    type_widths: HashMap<Word, u32>,
    /// Type classes (Bool/Int/Float/Other) keyed by type ID
    type_classes: HashMap<Word, TypeClass>,
    /// All root IDs (instructions we're optimizing)
    pub root_ids: Vec<Word>,
    /// GLSL.std.450 extended instruction set ID (if any)
    glsl_ext_id: Option<Word>,
    /// IDs of instructions that will be added to the egraph.
    /// Used by get_or_create_term to emit variable references (id{N})
    /// instead of Sym constructors for forward cross-block references.
    known_instruction_ids: HashSet<Word>,
    /// Additional egglog facts to run after all terms are bound.
    /// Used for seeding metadata like ResultWidth.
    pub additional_facts: Vec<String>,
}

impl EgglogContext {
    pub fn new(type_widths: &HashMap<Word, u32>, type_classes: &HashMap<Word, TypeClass>) -> Self {
        Self {
            id_to_term: HashMap::new(),
            id_to_type: HashMap::new(),
            type_widths: type_widths.clone(),
            type_classes: type_classes.clone(),
            root_ids: Vec::new(),
            glsl_ext_id: None,
            known_instruction_ids: HashSet::new(),
            additional_facts: Vec::new(),
        }
    }

    /// Pre-register an instruction so its ID and type are known before term creation.
    /// This allows `get_or_create_term` to emit variable references instead of Sym
    /// constructors for cross-block forward references.
    pub fn pre_register(&mut self, inst: &Instruction) {
        if let Some(id) = inst.result_id {
            if let Some(ty) = inst.result_type {
                self.id_to_type.insert(id, ty);
            }
            if super::is_optimizable(inst) {
                self.known_instruction_ids.insert(id);
            }
        }
    }

    /// Set the GLSL.std.450 extended instruction set ID.
    pub fn set_glsl_ext_id(&mut self, id: Word) {
        self.glsl_ext_id = Some(id);
    }

    /// Add an instruction to the context.
    pub fn add_instruction(&mut self, inst: &Instruction) {
        if let Some(result_id) = inst.result_id {
            if let Some(result_type) = inst.result_type {
                self.id_to_type.insert(result_id, result_type);
            }

            if let Some(term) = self.instruction_to_term(inst) {
                self.id_to_term.insert(result_id, term.clone());
                // Constants are NOT roots - they're only live if referenced by a root
                // This enables DCE to remove unused constants naturally through e-graph extraction
                if !matches!(
                    inst.class.opcode,
                    Op::Constant | Op::ConstantTrue | Op::ConstantFalse
                ) {
                    self.root_ids.push(result_id);
                }
            } else {
                // Term generation failed (e.g. Phi with different incoming values).
                // Remove from known_instruction_ids so get_or_create_term() falls
                // back to a Sym constructor instead of emitting a bare variable
                // reference that would be unbound in the egraph.
                self.known_instruction_ids.remove(&result_id);
                return;
            }

            // Seed ResultWidth for integer constants and conversions
            if let Some(result_type) = inst.result_type {
                let tc = self.type_class_of_type(result_type);
                if tc == TypeClass::Int {
                    let width = self.type_widths.get(&result_type).copied().unwrap_or(32);
                    match inst.class.opcode {
                        Op::Constant | Op::SConvert | Op::UConvert => {
                            self.additional_facts
                                .push(format!("(set (ResultWidth id{}) {})", result_id, width));
                        }
                        _ => {}
                    }
                }
            }

            // Detect same-type bitcast for redundant bitcast elimination
            if inst.class.opcode == Op::Bitcast {
                if let Some(result_type) = inst.result_type {
                    let src_id = inst.operands.iter().find_map(|op| op.id_ref_any());
                    if let Some(src_id) = src_id {
                        if let Some(src_type) = self.id_to_type.get(&src_id) {
                            if *src_type == result_type {
                                self.additional_facts
                                    .push(format!("(SameTypeBitcast id{})", result_id));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get the type class for a value ID (via its result type).
    fn type_class_of(&self, id: Word) -> TypeClass {
        self.id_to_type
            .get(&id)
            .and_then(|ty| self.type_classes.get(ty))
            .copied()
            .unwrap_or(TypeClass::Other)
    }

    /// Get the type class for a result type ID directly.
    fn type_class_of_type(&self, type_id: Word) -> TypeClass {
        self.type_classes
            .get(&type_id)
            .copied()
            .unwrap_or(TypeClass::Other)
    }

    /// Get or create a term for an operand ID.
    ///
    /// Returns a reference to the egglog variable for this ID.
    /// If the ID has a known term (was added via add_instruction), we reference
    /// the egglog variable directly. Otherwise, we use a typed Sym for external
    /// references (function parameters, globals, etc.).
    fn get_or_create_term(&mut self, id: Word) -> String {
        // Check if this ID is (or will be) bound in the e-graph.
        // We check known_instruction_ids in addition to id_to_term to handle
        // cross-block forward references where the instruction hasn't been
        // processed yet but will be (blocks may be out of dominance order).
        if self.id_to_term.contains_key(&id) || self.known_instruction_ids.contains(&id) {
            // Reference the egglog variable (will be bound via let)
            format!("id{}", id)
        } else {
            // External reference (function parameter, global, etc.)
            // Use typed Sym to place in the correct sort
            let sym_ctor = match self.type_class_of(id) {
                TypeClass::Int => "ISym",
                TypeClass::Float => "FSym",
                TypeClass::Bool => "BSym",
                TypeClass::Other => "Sym",
            };
            format!("({} \"id{}\")", sym_ctor, id)
        }
    }

    /// Convert an instruction to an egglog term.
    fn instruction_to_term(&mut self, inst: &Instruction) -> Option<String> {
        let term = match inst.class.opcode {
            Op::Constant => {
                let result_type = inst.result_type?;
                let width = self.type_widths.get(&result_type).copied().unwrap_or(32);
                let type_class = self.type_class_of_type(result_type);

                if type_class == TypeClass::Bool {
                    // Boolean constant via OpConstant (some compilers use this
                    // instead of OpConstantTrue/OpConstantFalse)
                    let value = inst.operands.iter().find_map(|op| match op {
                        Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })?;
                    format!("(BoolConst {})", if value != 0 { 1 } else { 0 })
                } else if type_class == TypeClass::Float {
                    // Float constant: reinterpret bits as IEEE float for FConst
                    let float_val: f64 = inst.operands.iter().find_map(|op| match op {
                        Operand::LiteralBit32(v) => Some(f32::from_bits(*v) as f64),
                        Operand::LiteralBit64(v) => Some(f64::from_bits(*v)),
                        _ => None,
                    })?;
                    // Format with decimal point for egglog f64 parsing
                    let s = format!("{}", float_val);
                    let literal = if s.contains('.')
                        || s.contains('e')
                        || s.contains('E')
                        || s == "inf"
                        || s == "-inf"
                        || s == "NaN"
                    {
                        s
                    } else {
                        format!("{}.0", s)
                    };
                    format!("(FConst {})", literal)
                } else {
                    // Integer constant: sign-extend 32-bit values so 0xFFFFFFFF becomes -1
                    // This enables rules like (BitXor x (Const -1)) to match
                    let value = inst.operands.iter().find_map(|op| match op {
                        Operand::LiteralBit32(v) => {
                            if width == 32 {
                                Some((*v as i32) as i64)
                            } else {
                                Some(*v as i64)
                            }
                        }
                        Operand::LiteralBit64(v) => Some(*v as i64),
                        _ => None,
                    })?;
                    if width == 64 {
                        format!("(Const64 {})", value)
                    } else {
                        format!("(Const {})", value)
                    }
                }
            }
            Op::ConstantTrue => "(BoolConst 1)".to_string(),
            Op::ConstantFalse => "(BoolConst 0)".to_string(),

            Op::IAdd => self.typed_binary_op("Add", "VecAdd", inst)?,
            Op::ISub => self.typed_binary_op("Sub", "VecSub", inst)?,
            Op::IMul => self.typed_binary_op("Mul", "VecMul", inst)?,
            Op::SDiv => self.typed_binary_op("SDiv", "VecSDiv", inst)?,
            Op::UDiv => self.typed_binary_op("UDiv", "VecUDiv", inst)?,
            Op::SRem => self.typed_binary_op("SRem", "VecSRem", inst)?,
            Op::SMod => self.typed_binary_op("SMod", "VecSMod", inst)?,
            Op::UMod => self.typed_binary_op("UMod", "VecUMod", inst)?,
            Op::SNegate => self.typed_unary_op("Neg", "VecNeg", inst)?,
            Op::ShiftLeftLogical => self.binary_op("Shl", inst)?,
            Op::ShiftRightLogical => self.binary_op("ShrU", inst)?,
            Op::ShiftRightArithmetic => self.binary_op("ShrS", inst)?,
            Op::BitwiseAnd => self.binary_op("BitAnd", inst)?,
            Op::BitwiseOr => self.binary_op("BitOr", inst)?,
            Op::BitwiseXor => self.binary_op("BitXor", inst)?,
            Op::Not => self.unary_op("BitNot", inst)?,
            Op::BitReverse => self.unary_op("BitReverse", inst)?,
            Op::IEqual => self.binary_op("Eq", inst)?,
            Op::INotEqual => self.binary_op("Ne", inst)?,
            Op::SLessThan => self.binary_op("SLt", inst)?,
            Op::SLessThanEqual => self.binary_op("SLe", inst)?,
            Op::SGreaterThan => self.binary_op("SGt", inst)?,
            Op::SGreaterThanEqual => self.binary_op("SGe", inst)?,
            Op::ULessThan => self.binary_op("ULt", inst)?,
            Op::ULessThanEqual => self.binary_op("ULe", inst)?,
            Op::UGreaterThan => self.binary_op("UGt", inst)?,
            Op::UGreaterThanEqual => self.binary_op("UGe", inst)?,
            Op::LogicalNot => self.unary_op("LogNot", inst)?,
            Op::LogicalAnd => self.binary_op("LogAnd", inst)?,
            Op::LogicalOr => self.binary_op("LogOr", inst)?,
            Op::LogicalEqual => self.binary_op("LogEq", inst)?,
            Op::LogicalNotEqual => self.binary_op("LogNe", inst)?,
            // Floating-point operations (scalar or vector)
            Op::FAdd => self.typed_binary_op("FAdd", "VecFAdd", inst)?,
            Op::FSub => self.typed_binary_op("FSub", "VecFSub", inst)?,
            Op::FMul => self.typed_binary_op("FMul", "VecFMul", inst)?,
            Op::FDiv => self.typed_binary_op("FDiv", "VecFDiv", inst)?,
            Op::FRem => self.typed_binary_op("FRem", "VecFRem", inst)?,
            Op::FMod => self.typed_binary_op("FMod", "VecFMod", inst)?,
            Op::FNegate => self.typed_unary_op("FNeg", "VecFNeg", inst)?,
            // Floating-point comparisons (ordered)
            Op::FOrdEqual => self.binary_op("FOrdEq", inst)?,
            Op::FOrdNotEqual => self.binary_op("FOrdNe", inst)?,
            Op::FOrdLessThan => self.binary_op("FOrdLt", inst)?,
            Op::FOrdLessThanEqual => self.binary_op("FOrdLe", inst)?,
            Op::FOrdGreaterThan => self.binary_op("FOrdGt", inst)?,
            Op::FOrdGreaterThanEqual => self.binary_op("FOrdGe", inst)?,
            // Floating-point comparisons (unordered)
            Op::FUnordEqual => self.binary_op("FUnordEq", inst)?,
            Op::FUnordNotEqual => self.binary_op("FUnordNe", inst)?,
            Op::FUnordLessThan => self.binary_op("FUnordLt", inst)?,
            Op::FUnordLessThanEqual => self.binary_op("FUnordLe", inst)?,
            Op::FUnordGreaterThan => self.binary_op("FUnordGt", inst)?,
            Op::FUnordGreaterThanEqual => self.binary_op("FUnordGe", inst)?,
            // Conversion operations
            Op::ConvertFToU => self.unary_op("ConvertFToU", inst)?,
            Op::ConvertFToS => self.unary_op("ConvertFToS", inst)?,
            Op::ConvertSToF => self.unary_op("ConvertSToF", inst)?,
            Op::ConvertUToF => self.unary_op("ConvertUToF", inst)?,
            Op::Select => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 3 {
                    let cond = self.get_or_create_term(ops[0]);
                    let t = self.get_or_create_term(ops[1]);
                    let f = self.get_or_create_term(ops[2]);
                    // Check if condition is a vector (TypeClass::Other = non-scalar)
                    let cond_is_vector = self.type_class_of(ops[0]) == TypeClass::Other;
                    if cond_is_vector {
                        // Vector select: component-wise, all operands are Expr
                        format!("(VecSelect {} {} {})", cond, t, f)
                    } else {
                        // Scalar select: use typed Select based on result type
                        let select_ctor =
                            match inst.result_type.map(|ty| self.type_class_of_type(ty)) {
                                Some(TypeClass::Int) => "SelectI",
                                Some(TypeClass::Float) => "SelectF",
                                Some(TypeClass::Bool) => "SelectB",
                                _ => "Select",
                            };
                        format!("({} {} {} {})", select_ctor, cond, t, f)
                    }
                } else {
                    return None;
                }
            }
            Op::CopyObject => {
                // CopyObject is just a reference to another value
                let operand_id = inst.operands.iter().find_map(|op| op.id_ref_any())?;
                self.get_or_create_term(operand_id)
            }
            // Derivative operations (fragment shader)
            Op::DPdx => self.unary_op("DPdx", inst)?,
            Op::DPdy => self.unary_op("DPdy", inst)?,
            Op::Fwidth => self.unary_op("Fwidth", inst)?,
            Op::DPdxFine => self.unary_op("DPdxFine", inst)?,
            Op::DPdyFine => self.unary_op("DPdyFine", inst)?,
            Op::FwidthFine => self.unary_op("FwidthFine", inst)?,
            Op::DPdxCoarse => self.unary_op("DPdxCoarse", inst)?,
            Op::DPdyCoarse => self.unary_op("DPdyCoarse", inst)?,
            Op::FwidthCoarse => self.unary_op("FwidthCoarse", inst)?,
            // Composite operations
            Op::CompositeExtract => {
                // CompositeExtract %type %composite indices...
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                let indices: Vec<u32> = inst
                    .operands
                    .iter()
                    .filter_map(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                if !ops.is_empty() && !indices.is_empty() {
                    let composite = self.get_or_create_term(ops[0]);
                    // For now, only handle single-level extraction (most common case)
                    format!("(CompositeExtract {} {})", composite, indices[0])
                } else {
                    return None;
                }
            }
            Op::CompositeInsert => {
                // CompositeInsert %type %object %composite indices...
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                let indices: Vec<u32> = inst
                    .operands
                    .iter()
                    .filter_map(|op| match op {
                        rspirv::dr::Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                if ops.len() >= 2 && !indices.is_empty() {
                    let object = self.get_or_create_term(ops[0]);
                    let composite = self.get_or_create_term(ops[1]);
                    format!("(CompositeInsert {} {} {})", composite, object, indices[0])
                } else {
                    return None;
                }
            }
            Op::CompositeConstruct => {
                // CompositeConstruct %type constituents...
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.is_empty() {
                    return None;
                }
                // Build ExprList for components
                let mut expr_list = "(ENil)".to_string();
                for &op_id in ops.iter().rev() {
                    let term = self.get_or_create_term(op_id);
                    expr_list = format!("(ECons {} {})", term, expr_list);
                }
                format!("(CompositeConstruct {})", expr_list)
            }
            Op::VectorExtractDynamic => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let vec = self.get_or_create_term(ops[0]);
                    let idx = self.get_or_create_term(ops[1]);
                    format!("(VectorExtractDynamic {} {})", vec, idx)
                } else {
                    return None;
                }
            }
            Op::VectorInsertDynamic => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 3 {
                    let vec = self.get_or_create_term(ops[0]);
                    let component = self.get_or_create_term(ops[1]);
                    let idx = self.get_or_create_term(ops[2]);
                    format!("(VectorInsertDynamic {} {} {})", vec, component, idx)
                } else {
                    return None;
                }
            }
            Op::VectorShuffle => {
                // VectorShuffle %type %vec1 %vec2 indices...
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                let indices: Vec<i64> = inst
                    .operands
                    .iter()
                    .filter_map(|op| match op {
                        Operand::LiteralBit32(v) => Some(*v as i64),
                        _ => None,
                    })
                    .collect();
                if ops.len() >= 2 {
                    let v1 = self.get_or_create_term(ops[0]);
                    let v2 = self.get_or_create_term(ops[1]);
                    match indices.len() {
                        2 => format!("(VecShuffle2 {} {} {} {})", v1, v2, indices[0], indices[1]),
                        3 => format!(
                            "(VecShuffle3 {} {} {} {} {})",
                            v1, v2, indices[0], indices[1], indices[2]
                        ),
                        4 => format!(
                            "(VecShuffle4 {} {} {} {} {} {})",
                            v1, v2, indices[0], indices[1], indices[2], indices[3]
                        ),
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            // Additional type conversions
            Op::SConvert => self.unary_op("SConvert", inst)?,
            Op::UConvert => self.unary_op("UConvert", inst)?,
            Op::FConvert => self.unary_op("FConvert", inst)?,
            Op::Bitcast => self.unary_op("Bitcast", inst)?,
            Op::QuantizeToF16 => self.unary_op("QuantizeToF16", inst)?,
            // FP predicates
            Op::IsNan => self.unary_op("IsNan", inst)?,
            Op::IsInf => self.unary_op("IsInf", inst)?,
            // Dot product
            Op::Dot => self.binary_op("Dot", inst)?,
            // Matrix operations
            Op::MatrixTimesScalar => self.binary_op("MatTimesScalar", inst)?,
            Op::MatrixTimesVector => self.binary_op("MatTimesVec", inst)?,
            Op::VectorTimesMatrix => self.binary_op("VecTimesMat", inst)?,
            Op::MatrixTimesMatrix => self.binary_op("MatTimesMat", inst)?,
            Op::Transpose => self.unary_op("Transpose", inst)?,
            Op::OuterProduct => self.binary_op("OuterProduct", inst)?,
            // Bit counting
            Op::BitCount => self.unary_op("BitCount", inst)?,
            // Extended instructions (GLSL.std.450)
            Op::ExtInst => self.extended_instruction_to_term(inst)?,
            // Memory operations - model in e-graph for load-store forwarding and dead store elimination
            Op::Load => {
                // Load %type %pointer [memory_access]
                let ptr_id = inst.operands.iter().find_map(|op| op.id_ref_any())?;
                let ptr = self.get_or_create_term(ptr_id);
                // Use InitMem as the memory state - real memory threading would need more work
                format!("(Load {} (InitMem))", ptr)
            }
            Op::Store => {
                // Store %pointer %object [memory_access]
                // Store has no result, but we model it for dead store elimination
                // We don't add it to the e-graph directly - stores are handled at block level
                return None;
            }
            Op::AccessChain => {
                // AccessChain %type %base indices...
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.is_empty() {
                    return None;
                }
                let base = self.get_or_create_term(ops[0]);
                // Get literal indices
                let indices: Vec<u32> = inst
                    .operands
                    .iter()
                    .skip(1)
                    .filter_map(|op| match op {
                        Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                // Also check for ID references as indices (dynamic access)
                let id_indices: Vec<Word> = ops.iter().skip(1).copied().collect();

                if !indices.is_empty() {
                    // Static indices - use AccessChain1/2/3
                    match indices.len() {
                        1 => format!("(AccessChain1 {} {})", base, indices[0]),
                        2 => format!("(AccessChain2 {} {} {})", base, indices[0], indices[1]),
                        3 => format!(
                            "(AccessChain3 {} {} {} {})",
                            base, indices[0], indices[1], indices[2]
                        ),
                        _ => return None,
                    }
                } else if !id_indices.is_empty() {
                    // Dynamic index - use AccessChainDyn
                    let idx = self.get_or_create_term(id_indices[0]);
                    format!("(AccessChainDyn {} {})", base, idx)
                } else {
                    return None;
                }
            }
            Op::InBoundsAccessChain => {
                // Same as AccessChain for optimization purposes
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.is_empty() {
                    return None;
                }
                let base = self.get_or_create_term(ops[0]);
                let indices: Vec<u32> = inst
                    .operands
                    .iter()
                    .skip(1)
                    .filter_map(|op| match op {
                        Operand::LiteralBit32(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                let id_indices: Vec<Word> = ops.iter().skip(1).copied().collect();

                if !indices.is_empty() {
                    match indices.len() {
                        1 => format!("(AccessChain1 {} {})", base, indices[0]),
                        2 => format!("(AccessChain2 {} {} {})", base, indices[0], indices[1]),
                        3 => format!(
                            "(AccessChain3 {} {} {} {})",
                            base, indices[0], indices[1], indices[2]
                        ),
                        _ => return None,
                    }
                } else if !id_indices.is_empty() {
                    let idx = self.get_or_create_term(id_indices[0]);
                    format!("(AccessChainDyn {} {})", base, idx)
                } else {
                    return None;
                }
            }
            Op::Variable => {
                // Variable %type storage_class [initializer]
                // Model as Var node with storage class
                let storage_class = inst
                    .operands
                    .iter()
                    .find_map(|op| match op {
                        Operand::StorageClass(sc) => Some(*sc as i64),
                        _ => None,
                    })
                    .unwrap_or(0);
                let name = inst
                    .result_id
                    .map(|id| format!("var_{}", id))
                    .unwrap_or_default();
                format!("(Var \"{}\" {})", name, storage_class)
            }
            // Image operations - model for texture hoisting and CSE
            Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod => {
                // ImageSample* %type %sampled_image %coordinate [ImageOperands offset_id...]
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let img = self.get_or_create_term(ops[0]);
                    let coord = self.get_or_create_term(ops[1]);
                    // Check for Offset-only or ConstOffset-only image operands
                    let mask = inst.operands.iter().find_map(|op| match op {
                        Operand::ImageOperands(m) => Some(*m),
                        _ => None,
                    });
                    match mask {
                        Some(m) if m == rspirv::spirv::ImageOperands::OFFSET => {
                            if ops.len() >= 3 {
                                let off = self.get_or_create_term(ops[2]);
                                format!("(ImageSampleOffset {} {} {})", img, coord, off)
                            } else {
                                format!("(ImageSample {} {})", img, coord)
                            }
                        }
                        Some(m) if m == rspirv::spirv::ImageOperands::CONST_OFFSET => {
                            if ops.len() >= 3 {
                                let off = self.get_or_create_term(ops[2]);
                                format!("(ImageSampleConstOffset {} {} {})", img, coord, off)
                            } else {
                                format!("(ImageSample {} {})", img, coord)
                            }
                        }
                        _ => format!("(ImageSample {} {})", img, coord),
                    }
                } else {
                    return None;
                }
            }
            Op::ImageFetch => {
                // ImageFetch %type %image %coordinate [ImageOperands offset_id...]
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let img = self.get_or_create_term(ops[0]);
                    let coord = self.get_or_create_term(ops[1]);
                    // Check for Offset-only or ConstOffset-only image operands
                    let mask = inst.operands.iter().find_map(|op| match op {
                        Operand::ImageOperands(m) => Some(*m),
                        _ => None,
                    });
                    match mask {
                        Some(m) if m == rspirv::spirv::ImageOperands::OFFSET => {
                            if ops.len() >= 3 {
                                let off = self.get_or_create_term(ops[2]);
                                format!("(ImageFetchOffset {} {} {})", img, coord, off)
                            } else {
                                format!("(ImageFetch {} {})", img, coord)
                            }
                        }
                        Some(m) if m == rspirv::spirv::ImageOperands::CONST_OFFSET => {
                            if ops.len() >= 3 {
                                let off = self.get_or_create_term(ops[2]);
                                format!("(ImageFetchConstOffset {} {} {})", img, coord, off)
                            } else {
                                format!("(ImageFetch {} {})", img, coord)
                            }
                        }
                        _ => format!("(ImageFetch {} {})", img, coord),
                    }
                } else {
                    return None;
                }
            }
            Op::ImageRead => {
                // ImageRead %type %image %coordinate [operands]
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let img = self.get_or_create_term(ops[0]);
                    let coord = self.get_or_create_term(ops[1]);
                    format!("(ImageRead {} {})", img, coord)
                } else {
                    return None;
                }
            }
            Op::SampledImage => {
                // SampledImage %type %image %sampler -> combined image+sampler
                // Model as binary operation for CSE
                self.binary_op("SampledImage", inst)?
            }
            Op::Image => {
                // Image %type %sampled_image -> extract image from combined
                self.unary_op("Image", inst)?
            }
            // Atomic operations - model for optimization across atomics
            Op::AtomicLoad => {
                // AtomicLoad %type %pointer %scope %semantics
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if !ops.is_empty() {
                    let ptr = self.get_or_create_term(ops[0]);
                    format!("(AtomicLoad {} (InitMem))", ptr)
                } else {
                    return None;
                }
            }
            Op::AtomicStore => {
                // AtomicStore has no result - skip
                return None;
            }
            Op::AtomicExchange => {
                // AtomicExchange %type %pointer %scope %semantics %value
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let ptr = self.get_or_create_term(ops[0]);
                    // Value is typically the last ID operand
                    let val = self.get_or_create_term(*ops.last().unwrap());
                    format!("(AtomicExchange {} {} (InitMem))", ptr, val)
                } else {
                    return None;
                }
            }
            Op::AtomicCompareExchange => {
                // AtomicCompareExchange %type %ptr %scope %eq_sem %neq_sem %value %comparator
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 3 {
                    let ptr = self.get_or_create_term(ops[0]);
                    let val = self.get_or_create_term(ops[ops.len() - 2]);
                    let cmp = self.get_or_create_term(ops[ops.len() - 1]);
                    format!("(AtomicCompareExchange {} {} {} (InitMem))", ptr, cmp, val)
                } else {
                    return None;
                }
            }
            Op::AtomicIAdd => self.atomic_binary_op("AtomicIAdd", inst)?,
            Op::AtomicISub => self.atomic_binary_op("AtomicISub", inst)?,
            Op::AtomicSMin => self.atomic_binary_op("AtomicSMin", inst)?,
            Op::AtomicUMin => self.atomic_binary_op("AtomicUMin", inst)?,
            Op::AtomicSMax => self.atomic_binary_op("AtomicSMax", inst)?,
            Op::AtomicUMax => self.atomic_binary_op("AtomicUMax", inst)?,
            Op::AtomicAnd => self.atomic_binary_op("AtomicAnd", inst)?,
            Op::AtomicOr => self.atomic_binary_op("AtomicOr", inst)?,
            Op::AtomicXor => self.atomic_binary_op("AtomicXor", inst)?,
            // Barrier operations - model for barrier hoisting/merging
            Op::ControlBarrier => {
                // ControlBarrier %execution_scope %memory_scope %semantics
                // Model as effect for barrier motion optimization
                // Barriers have no result, but we track them for control flow
                return None;
            }
            Op::MemoryBarrier => {
                // MemoryBarrier %memory_scope %semantics
                return None;
            }
            // Subgroup operations - model for CSE and optimization
            Op::GroupNonUniformElect => {
                // GroupNonUniformElect %type %scope -> bool (true for first invocation)
                "(GroupElect)".to_string()
            }
            Op::GroupNonUniformAll => {
                // GroupNonUniformAll %type %scope %predicate
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if !ops.is_empty() {
                    let pred = self.get_or_create_term(*ops.last().unwrap());
                    format!("(GroupAll {})", pred)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformAny => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if !ops.is_empty() {
                    let pred = self.get_or_create_term(*ops.last().unwrap());
                    format!("(GroupAny {})", pred)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformAllEqual => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if !ops.is_empty() {
                    let val = self.get_or_create_term(*ops.last().unwrap());
                    format!("(GroupAllEqual {})", val)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformBroadcast => {
                // GroupNonUniformBroadcast %type %scope %value %id
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let val = self.get_or_create_term(ops[ops.len() - 2]);
                    let id = self.get_or_create_term(ops[ops.len() - 1]);
                    format!("(GroupBroadcast {} {})", val, id)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformBroadcastFirst => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if !ops.is_empty() {
                    let val = self.get_or_create_term(*ops.last().unwrap());
                    format!("(GroupBroadcastFirst {})", val)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformShuffle => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let val = self.get_or_create_term(ops[ops.len() - 2]);
                    let id = self.get_or_create_term(ops[ops.len() - 1]);
                    format!("(GroupShuffle {} {})", val, id)
                } else {
                    return None;
                }
            }
            Op::GroupNonUniformShuffleXor => {
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                if ops.len() >= 2 {
                    let val = self.get_or_create_term(ops[ops.len() - 2]);
                    let mask = self.get_or_create_term(ops[ops.len() - 1]);
                    format!("(GroupShuffleXor {} {})", val, mask)
                } else {
                    return None;
                }
            }
            // Subgroup arithmetic reductions
            Op::GroupNonUniformIAdd => self.subgroup_reduction_op("GroupIAdd", inst)?,
            Op::GroupNonUniformFAdd => self.subgroup_reduction_op("GroupFAdd", inst)?,
            Op::GroupNonUniformIMul => self.subgroup_reduction_op("GroupIMul", inst)?,
            Op::GroupNonUniformFMul => self.subgroup_reduction_op("GroupFMul", inst)?,
            Op::GroupNonUniformSMin => self.subgroup_reduction_op("GroupSMin", inst)?,
            Op::GroupNonUniformUMin => self.subgroup_reduction_op("GroupUMin", inst)?,
            Op::GroupNonUniformFMin => self.subgroup_reduction_op("GroupFMin", inst)?,
            Op::GroupNonUniformSMax => self.subgroup_reduction_op("GroupSMax", inst)?,
            Op::GroupNonUniformUMax => self.subgroup_reduction_op("GroupUMax", inst)?,
            Op::GroupNonUniformFMax => self.subgroup_reduction_op("GroupFMax", inst)?,
            Op::GroupNonUniformBitwiseAnd => self.subgroup_reduction_op("GroupBitAnd", inst)?,
            Op::GroupNonUniformBitwiseOr => self.subgroup_reduction_op("GroupBitOr", inst)?,
            Op::GroupNonUniformBitwiseXor => self.subgroup_reduction_op("GroupBitXor", inst)?,
            Op::GroupNonUniformLogicalAnd => self.subgroup_reduction_op("GroupLogAnd", inst)?,
            Op::GroupNonUniformLogicalOr => self.subgroup_reduction_op("GroupLogOr", inst)?,
            Op::GroupNonUniformLogicalXor => self.subgroup_reduction_op("GroupLogXor", inst)?,
            Op::Phi => {
                // For Phi, check if all incoming values are the same
                // Phi operands are pairs: (value, block) repeated
                let ops: Vec<Word> = inst
                    .operands
                    .iter()
                    .filter_map(|op| op.id_ref_any())
                    .collect();
                // Extract just the values (even indices)
                let values: Vec<Word> = ops.iter().step_by(2).copied().collect();
                if values.is_empty() {
                    return None;
                }
                // If all values are the same, the phi simplifies to that value
                // Use a typed Sym reference so that after extraction we get a CopyObject
                let first = values[0];
                if values.iter().all(|&v| v == first) {
                    let sym_ctor = match inst.result_type.map(|ty| self.type_class_of_type(ty)) {
                        Some(TypeClass::Int) => "ISym",
                        Some(TypeClass::Float) => "FSym",
                        Some(TypeClass::Bool) => "BSym",
                        _ => "Sym",
                    };
                    format!("({} \"id{}\")", sym_ctor, first)
                } else {
                    // Can't optimize a phi with different values in the e-graph yet
                    return None;
                }
            }
            _ => return None,
        };

        Some(term)
    }

    fn binary_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        let ops: Vec<Word> = inst
            .operands
            .iter()
            .filter_map(|op| op.id_ref_any())
            .collect();
        if ops.len() >= 2 {
            let lhs = self.get_or_create_term(ops[0]);
            let rhs = self.get_or_create_term(ops[1]);
            Some(format!("({} {} {})", op, lhs, rhs))
        } else {
            None
        }
    }

    fn unary_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        let operand_id = inst.operands.iter().find_map(|op| op.id_ref_any())?;
        let operand = self.get_or_create_term(operand_id);
        Some(format!("({} {})", op, operand))
    }

    /// Type-dispatched binary op: uses `scalar_op` (typed sort) when the result
    /// type is a scalar, and `vec_op` (Expr sort) when it is a vector/other type.
    /// This prevents sort mismatches when SPIR-V opcodes like OpFAdd/OpIAdd
    /// operate on vectors whose operands are in the generic Expr sort.
    fn typed_binary_op(
        &mut self,
        scalar_op: &str,
        vec_op: &str,
        inst: &Instruction,
    ) -> Option<String> {
        let is_scalar = inst
            .result_type
            .map(|ty| self.type_class_of_type(ty) != TypeClass::Other)
            .unwrap_or(false);
        if is_scalar {
            self.binary_op(scalar_op, inst)
        } else {
            self.binary_op(vec_op, inst)
        }
    }

    /// Type-dispatched unary op: uses `scalar_op` (typed sort) when the result
    /// type is a scalar, and `vec_op` (Expr sort) when it is a vector/other type.
    fn typed_unary_op(
        &mut self,
        scalar_op: &str,
        vec_op: &str,
        inst: &Instruction,
    ) -> Option<String> {
        let is_scalar = inst
            .result_type
            .map(|ty| self.type_class_of_type(ty) != TypeClass::Other)
            .unwrap_or(false);
        if is_scalar {
            self.unary_op(scalar_op, inst)
        } else {
            self.unary_op(vec_op, inst)
        }
    }

    /// Convert GLSL.std.450 extended instruction to egglog term.
    fn extended_instruction_to_term(&mut self, inst: &Instruction) -> Option<String> {
        // ExtInst operands: %set %instruction operands...
        // operands[0] = extended instruction set ID
        // operands[1] = instruction number (LiteralExtInstInteger)
        // operands[2..] = instruction operands
        let set_id = inst.operands.first()?.id_ref_any()?;

        // Check if this is GLSL.std.450
        if self.glsl_ext_id != Some(set_id) {
            return None;
        }

        let ext_opcode = match &inst.operands.get(1)? {
            Operand::LiteralExtInstInteger(n) => *n,
            _ => return None,
        };

        // Get operand IDs (skip set ID and opcode)
        let op_ids: Vec<Word> = inst
            .operands
            .iter()
            .skip(2)
            .filter_map(|op| op.id_ref_any())
            .collect();

        // GLSL.std.450 instruction numbers
        // See: https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
        match ext_opcode {
            // Trigonometric
            13 => self.ext_unary("Sin", &op_ids),    // Sin
            14 => self.ext_unary("Cos", &op_ids),    // Cos
            15 => self.ext_unary("Tan", &op_ids),    // Tan
            16 => self.ext_unary("Asin", &op_ids),   // Asin
            17 => self.ext_unary("Acos", &op_ids),   // Acos
            18 => self.ext_unary("Atan", &op_ids),   // Atan
            19 => self.ext_unary("Sinh", &op_ids),   // Sinh
            20 => self.ext_unary("Cosh", &op_ids),   // Cosh
            21 => self.ext_unary("Tanh", &op_ids),   // Tanh
            22 => self.ext_unary("Asinh", &op_ids),  // Asinh
            23 => self.ext_unary("Acosh", &op_ids),  // Acosh
            24 => self.ext_unary("Atanh", &op_ids),  // Atanh
            25 => self.ext_binary("Atan2", &op_ids), // Atan2

            // Exponential
            26 => self.ext_binary("Pow", &op_ids), // Pow
            27 => self.ext_unary("Exp", &op_ids),  // Exp
            28 => self.ext_unary("Log", &op_ids),  // Log
            29 => self.ext_unary("Exp2", &op_ids), // Exp2
            30 => self.ext_unary("Log2", &op_ids), // Log2
            31 => self.ext_unary("Sqrt", &op_ids), // Sqrt
            32 => self.ext_unary("InverseSqrt", &op_ids), // InverseSqrt

            // Common
            33 => self.ext_unary("Determinant", &op_ids), // Determinant
            34 => self.ext_unary("MatInverse", &op_ids),  // MatrixInverse

            // Modf/Frexp with struct return
            35 => self.ext_unary("ModfStruct", &op_ids), // ModfStruct
            36 => self.ext_unary("Modf", &op_ids),       // Modf (returns fraction)
            51 => self.ext_unary("FrexpStruct", &op_ids), // FrexpStruct
            52 => self.ext_unary("Frexp", &op_ids),      // Frexp (returns sig)
            53 => self.ext_binary("Ldexp", &op_ids),     // Ldexp

            // Pack/Unpack
            54 => self.ext_unary("PackSnorm4x8", &op_ids),
            55 => self.ext_unary("PackUnorm4x8", &op_ids),
            56 => self.ext_unary("PackSnorm2x16", &op_ids),
            57 => self.ext_unary("PackUnorm2x16", &op_ids),
            58 => self.ext_unary("PackHalf2x16", &op_ids),
            59 => self.ext_unary("PackDouble2x32", &op_ids),
            60 => self.ext_unary("UnpackSnorm2x16", &op_ids),
            61 => self.ext_unary("UnpackUnorm2x16", &op_ids),
            62 => self.ext_unary("UnpackHalf2x16", &op_ids),
            63 => self.ext_unary("UnpackSnorm4x8", &op_ids),
            64 => self.ext_unary("UnpackUnorm4x8", &op_ids),
            65 => self.ext_unary("UnpackDouble2x32", &op_ids),

            // Length/Distance/Cross
            66 => self.ext_unary("Length", &op_ids), // Length
            67 => self.ext_binary("Distance", &op_ids), // Distance
            68 => self.ext_binary("Cross", &op_ids), // Cross
            69 => self.ext_unary("Normalize", &op_ids), // Normalize
            70 => self.ext_ternary("FaceForward", &op_ids), // FaceForward
            71 => self.ext_binary("Reflect", &op_ids), // Reflect
            72 => self.ext_ternary("Refract", &op_ids), // Refract

            // Integer bit manipulation
            73 => self.ext_unary("FindILsb", &op_ids), // FindILsb
            74 => self.ext_unary("FindSMsb", &op_ids), // FindSMsb
            75 => self.ext_unary("FindUMsb", &op_ids), // FindUMsb

            // Abs/Sign
            4 => self.ext_unary("FAbs", &op_ids),  // FAbs
            5 => self.ext_unary("SAbs", &op_ids),  // SAbs
            6 => self.ext_unary("FSign", &op_ids), // FSign
            7 => self.ext_unary("Sign", &op_ids),  // SSign

            // Floor/Ceil/Round/Trunc/Fract
            8 => self.ext_unary("FFloor", &op_ids),   // Floor
            9 => self.ext_unary("FCeil", &op_ids),    // Ceil
            10 => self.ext_unary("Fract", &op_ids),   // Fract
            11 => self.ext_unary("Radians", &op_ids), // Radians
            12 => self.ext_unary("Degrees", &op_ids), // Degrees

            // Round/Trunc
            1 => self.ext_unary("FRound", &op_ids), // Round
            2 => self.ext_unary("FRound", &op_ids), // RoundEven (same as Round for now)
            3 => self.ext_unary("FTrunc", &op_ids), // Trunc

            // Min/Max/Clamp (GLSL.std.450 opcodes)
            37 => self.ext_binary("FMin", &op_ids),    // FMin
            38 => self.ext_binary("UMin", &op_ids),    // UMin
            39 => self.ext_binary("SMin", &op_ids),    // SMin
            40 => self.ext_binary("FMax", &op_ids),    // FMax
            41 => self.ext_binary("UMax", &op_ids),    // UMax
            42 => self.ext_binary("SMax", &op_ids),    // SMax
            43 => self.ext_ternary("FClamp", &op_ids), // FClamp
            44 => self.ext_ternary("UClamp", &op_ids), // UClamp
            45 => self.ext_ternary("SClamp", &op_ids), // SClamp
            46 => {
                // FMix: scalar float → FMix(FloatExpr), vector → VecFMix(Expr)
                let is_scalar = inst
                    .result_type
                    .map(|ty| self.type_class_of_type(ty) == TypeClass::Float)
                    .unwrap_or(false);
                if is_scalar {
                    self.ext_ternary("FMix", &op_ids)
                } else {
                    self.ext_ternary("VecFMix", &op_ids)
                }
            }

            // Step/SmoothStep
            48 => self.ext_binary("Step", &op_ids), // Step
            49 => self.ext_ternary("SmoothStep", &op_ids), // SmoothStep

            // Fma
            50 => self.ext_ternary("Fma", &op_ids), // Fma

            // NMin/NMax/NClamp
            79 => self.ext_binary("NMin", &op_ids),    // NMin
            80 => self.ext_binary("NMax", &op_ids),    // NMax
            81 => self.ext_ternary("NClamp", &op_ids), // NClamp

            _ => None,
        }
    }

    fn ext_unary(&mut self, op: &str, op_ids: &[Word]) -> Option<String> {
        if !op_ids.is_empty() {
            let a = self.get_or_create_term(op_ids[0]);
            Some(format!("({} {})", op, a))
        } else {
            None
        }
    }

    fn ext_binary(&mut self, op: &str, op_ids: &[Word]) -> Option<String> {
        if op_ids.len() >= 2 {
            let a = self.get_or_create_term(op_ids[0]);
            let b = self.get_or_create_term(op_ids[1]);
            Some(format!("({} {} {})", op, a, b))
        } else {
            None
        }
    }

    fn ext_ternary(&mut self, op: &str, op_ids: &[Word]) -> Option<String> {
        if op_ids.len() >= 3 {
            let a = self.get_or_create_term(op_ids[0]);
            let b = self.get_or_create_term(op_ids[1]);
            let c = self.get_or_create_term(op_ids[2]);
            Some(format!("({} {} {} {})", op, a, b, c))
        } else {
            None
        }
    }

    /// Convert atomic binary operations (ptr, value) to e-graph term.
    fn atomic_binary_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        // Atomic binary ops: %type %ptr %scope %semantics %value
        let ops: Vec<Word> = inst
            .operands
            .iter()
            .filter_map(|op| op.id_ref_any())
            .collect();
        if ops.len() >= 2 {
            let ptr = self.get_or_create_term(ops[0]);
            // Value is the last ID operand
            let val = self.get_or_create_term(*ops.last().unwrap());
            Some(format!("({} {} {} (InitMem))", op, ptr, val))
        } else {
            None
        }
    }

    /// Convert subgroup reduction operations to e-graph term.
    fn subgroup_reduction_op(&mut self, op: &str, inst: &Instruction) -> Option<String> {
        // GroupNonUniform* %type %scope %operation %value [cluster_size]
        let ops: Vec<Word> = inst
            .operands
            .iter()
            .filter_map(|op| op.id_ref_any())
            .collect();
        if !ops.is_empty() {
            // Value is the last ID operand
            let val = self.get_or_create_term(*ops.last().unwrap());
            Some(format!("({} {})", op, val))
        } else {
            None
        }
    }
}
