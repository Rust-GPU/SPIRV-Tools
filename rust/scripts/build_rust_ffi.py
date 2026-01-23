#!/usr/bin/env python3
# Copyright 2024 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the spirv-tools-ffi Rust crate")
    parser.add_argument("--manifest-path", required=True, help="Path to Cargo.toml for the workspace")
    parser.add_argument("--profile", default="release", help="Cargo profile to build (release/debug/custom)")
    parser.add_argument("--target-dir", required=True, help="Directory to use for CARGO_TARGET_DIR")
    parser.add_argument("--output", required=True, help="Location to copy the final static library")
    parser.add_argument("--package", default="spirv-tools-ffi", help="Cargo package to build")
    return parser.parse_args()


def profile_args(profile: str) -> tuple[list[str], str]:
    normalized = profile.strip()
    if normalized == "release":
        return (["--release"], "release")
    if normalized == "debug":
        return ([], "debug")
    return (["--profile", normalized], normalized)


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing Cargo manifest: {manifest}")

    cargo_args, profile_dir = profile_args(args.profile)
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    # Skip linking C++ SPIRV-Tools libraries - Bazel will link them at final link time
    env["SPIRV_TOOLS_FFI_SKIP_CPP_LINK"] = "1"
    # Use Rust-only implementations in context_bridge.cc (no dependency on generated headers)
    env["SPIRV_RUST_TARGET_ENV_DEFINE"] = "1"

    package = args.package
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "-p",
        package,
    ] + cargo_args

    workdir = manifest.parent
    subprocess.run(command, cwd=workdir, check=True, env=env)

    lib_name = f"lib{package.replace('-', '_')}.a"
    built_lib = target_dir / profile_dir / lib_name
    if not built_lib.exists():
        raise FileNotFoundError(f"cargo did not produce {built_lib}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_lib, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
