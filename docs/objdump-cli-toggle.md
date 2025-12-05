# Rust `spirv-objdump` Toggles

The Rust `spirv-objdump` binary is a drop-in replacement for the legacy C++ tool. By default it preserves parity, including treating `--compiler-cmd` as unimplemented. An opt-in toggle exposes the recorded compiler command when available.

## Environment

- `SPIRV_TOOLS_ENABLE_COMPILER_CMD=1` — enable extraction of compiler commands recorded via `OpModuleProcessed`. When set, `--compiler-cmd` prints the concatenated strings from each `OpModuleProcessed`. Without this flag the option remains unimplemented to match the C++ tool.

## Behavior

- `--source`/`--list`/`--outdir` mirror the C++ tool (export messages, skip empty sources, overwrite guard with `--force`).
- `--compiler-cmd` fails with `unimplemented` unless `SPIRV_TOOLS_ENABLE_COMPILER_CMD` is set. When enabled, it returns a typed error if no compiler command is recorded.

## Rollout Notes

- Keep the env toggle disabled in CI by default so help/usage parity remains intact.
- When comparing against the C++ tool, unset `SPIRV_TOOLS_ENABLE_COMPILER_CMD` so exit codes and stderr remain aligned.
