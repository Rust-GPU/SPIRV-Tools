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

#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rust/cxxbridge/spirv-tools-ffi.h"

namespace spvtools {
namespace {

#if defined(SPIRV_RUST_TARGET_ENV)

void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
  if (value) {
    _putenv_s(name, value);
  } else {
    _putenv_s(name, "");
  }
#else
  if (value) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

TEST(RustOptimizerBridge, ReportsParseError) {
  spvtools::ffi::clear_rust_optimizer_override();
  const uint32_t bad_word = 0u;
  ::rust::Slice<const uint32_t> words(&bad_word, 1);
  const auto result = spvtools::ffi::optimize_basic_block(words);
  EXPECT_FALSE(result.success);
  EXPECT_EQ(spvtools::ffi::OptimizeError::Parse, result.error);
}

TEST(RustOptimizerBridge, ReportsDisabledKind) {
  spvtools::ffi::clear_rust_optimizer_override();
  SetEnv("SPIRV_TOOLS_DISABLE_RUST_OPT", "1");

  const uint32_t spirv[] = {
      0x07230203, 0x00010100, 0x00000000, 0x00000001, 0x00000000,
      0x00020011, 0x00000001, 0x00020011, 0x00000005, 0x0003000E,
      0x00000000, 0x00000001};

  ::rust::Slice<const uint32_t> words(spirv, sizeof(spirv) / sizeof(uint32_t));
  const auto result = spvtools::ffi::optimize_basic_block(words);
  SetEnv("SPIRV_TOOLS_DISABLE_RUST_OPT", nullptr);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(spvtools::ffi::OptimizeError::Disabled, result.error);
  EXPECT_THAT(result.words, ::testing::ElementsAreArray(spirv));
}

TEST(RustOptimizerBridge, OverrideTogglesOptimizer) {
  spvtools::ffi::clear_rust_optimizer_override();
  SetEnv("SPIRV_TOOLS_DISABLE_RUST_OPT", nullptr);

  const uint32_t spirv[] = {
      0x07230203, 0x00010100, 0x00000000, 0x00000001, 0x00000000,
      0x00020011, 0x00000001, 0x0003000E, 0x00000000, 0x00000001,
      0x0004002B, 0x00000006, 0x00000009, 0x00000002, 0x0004002B,
      0x00000006, 0x0000000A, 0x00000003, 0x00050080, 0x00000006,
      0x0000000B, 0x00000009, 0x0000000A, 0x000100FD};

  // Force disable via override.
  spvtools::ffi::set_rust_optimizer_override(false);
  ::rust::Slice<const uint32_t> words_disabled(spirv,
                                               sizeof(spirv) / sizeof(uint32_t));
  const auto disabled = spvtools::ffi::optimize_basic_block(words_disabled);
  EXPECT_TRUE(disabled.success);
  EXPECT_EQ(spvtools::ffi::OptimizeError::Disabled, disabled.error);

  // Force enable even if the disable env remains set.
  spvtools::ffi::set_rust_optimizer_override(true);
  ::rust::Slice<const uint32_t> words_enabled(spirv,
                                              sizeof(spirv) / sizeof(uint32_t));
  const auto enabled = spvtools::ffi::optimize_basic_block(words_enabled);
  EXPECT_TRUE(enabled.success);
  EXPECT_EQ(spvtools::ffi::OptimizeError::None, enabled.error);
  spvtools::ffi::clear_rust_optimizer_override();
  SetEnv("SPIRV_TOOLS_DISABLE_RUST_OPT", nullptr);
}

#else
TEST(RustOptimizerBridge, SkipWithoutRustTarget) { GTEST_SKIP(); }
#endif  // SPIRV_RUST_TARGET_ENV

}  // namespace
}  // namespace spvtools
