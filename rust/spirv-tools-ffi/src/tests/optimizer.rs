use super::super::optimizer::optimize_basic_block;

#[test]
fn optimizer_basic_block_pass_through_non_arith() {
    // A trivial module with OpMemoryModel only; optimizer should pass it through unchanged.
    let words = vec![
        0x07230203, // magic
        0x00010000, // version 1.0
        0,          // generator
        5,          // bound
        0,          // schema
        0x00020011, // OpMemoryModel Logical Simple
        0x00000001,
        0x00000000,
    ];
    let result = optimize_basic_block(&words).expect("optimization should succeed");
    assert_eq!(result, words);
}
