// Copyright (c) 2025 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rust/cxxbridge/spirv-tools-ffi.h"

namespace spvtools {
namespace {

#if defined(SPIRV_RUST_TARGET_ENV)
TEST(RustValidatorOverride, TogglePaths) {
  // Default-on when not overridden.
  spvtools::ffi::clear_rust_validator_override();
  EXPECT_TRUE(spvtools::ffi::rust_validator_enabled());

  spvtools::ffi::set_rust_validator_override(false);
  EXPECT_FALSE(spvtools::ffi::rust_validator_enabled());

  spvtools::ffi::set_rust_validator_override(true);
  EXPECT_TRUE(spvtools::ffi::rust_validator_enabled());

  spvtools::ffi::clear_rust_validator_override();
  EXPECT_TRUE(spvtools::ffi::rust_validator_enabled());
}
#else
TEST(RustValidatorOverride, SkipWithoutRustTarget) { GTEST_SKIP(); }
#endif  // SPIRV_RUST_TARGET_ENV

}  // namespace
}  // namespace spvtools

