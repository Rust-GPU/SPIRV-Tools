# Optimizer Parity Runner

Use `scripts/run-opt-parity.sh` to compare the Rust arithmetic optimizer with the C++ `spirv-opt`
binary over the parity corpus.

## Requirements
- `spirv-opt` in `PATH`, or set `SPIRV_CPP_OPT=/path/to/spirv-opt`.
- Build the Rust workspace (`cargo test -p spirv-tools-opt --tests`) to ensure dependencies are ready.

## Usage
```bash
scripts/run-opt-parity.sh               # auto-detects workspace root and spirv-opt
scripts/run-opt-parity.sh /path/to/ws   # explicit workspace root
SPIRV_CPP_OPT=/opt/bin/spirv-opt scripts/run-opt-parity.sh
```

The script runs the `cpp_parity` tests in `spirv-tools-opt` with the chosen C++ optimizer, failing if
outputs diverge. Integrate it into CI to gate regressions and track Rust/C++ optimizer parity.
