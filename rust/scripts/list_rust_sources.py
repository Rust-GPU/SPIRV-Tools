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
import os
from pathlib import Path

EXCLUDED_DIRS = {"target", ".git"}


def main() -> None:
    script_path = Path(__file__).resolve()
    rust_dir = script_path.parent.parent

    files: list[str] = []
    for root, dirs, filenames in os.walk(rust_dir):
        # Prune excluded directories in-place so os.walk does not descend.
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        if any(part in EXCLUDED_DIRS for part in Path(root).parts):
            continue
        for filename in filenames:
            candidate = Path(root) / filename
            if any(part in EXCLUDED_DIRS for part in candidate.parts):
                continue
            files.append(str(candidate.resolve()))

    for entry in sorted(set(files)):
        print(entry)


if __name__ == "__main__":
    main()
