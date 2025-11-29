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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "rust/cxxbridge/spirv-tools-ffi.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace {

using ::testing::Not;
using ::testing::StrEq;

#if defined(SPIRV_RUST_TARGET_ENV)
std::vector<uint32_t> AssembleOrDie(const std::string& text) {
  spv_target_env env = SPV_ENV_UNIVERSAL_1_6;
  SpirvTools tools(env);
  std::string message;
  tools.SetMessageConsumer(
      [&message](spv_message_level_t, const char*, const spv_position_t&,
                 const char* msg) { message = msg; });
  std::vector<uint32_t> binary;
  EXPECT_TRUE(tools.Assemble(text, &binary, kFuzzAssembleOption))
      << "assembler error: " << message;
  return binary;
}

struct ValidationResult {
  bool success;
  std::string message;
};

ValidationResult ValidateWithCpp(const std::vector<uint32_t>& words) {
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
  std::string message;
  tools.SetMessageConsumer(
      [&message](spv_message_level_t, const char*, const spv_position_t&,
                 const char* msg) { message = msg; });
  const bool success = tools.Validate(words);
  return ValidationResult{success, message};
}

ValidationResult ValidateWithRust(const std::vector<uint32_t>& words) {
  ffi::ValidatorOptions options{};
  const auto result = ffi::validate_binary_with_options(
      static_cast<uint32_t>(SPV_ENV_UNIVERSAL_1_6),
      ::rust::Slice<const uint32_t>(words.data(), words.size()), options);
  return ValidationResult{result.success,
                          static_cast<std::string>(result.message)};
}

TEST(RustValidatorParity, ValidModuleMatchesCpp) {
  const auto words = AssembleOrDie(
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)");

  const auto cpp = ValidateWithCpp(words);
  const auto rust = ValidateWithRust(words);
  EXPECT_TRUE(cpp.success);
  EXPECT_TRUE(rust.success);
  EXPECT_TRUE(rust.message.empty());
}

TEST(RustValidatorParity, InvalidModuleMatchesCpp) {
  const auto words = AssembleOrDie(
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%var = OpVariable %ptr Input
OpDecorate %var BuiltIn SubgroupSize
OpReturn
OpFunctionEnd
)");

  const auto cpp = ValidateWithCpp(words);
  const auto rust = ValidateWithRust(words);
  EXPECT_FALSE(cpp.success);
  EXPECT_FALSE(rust.success);
  EXPECT_THAT(rust.message, Not(StrEq("")));
  EXPECT_THAT(cpp.message, Not(StrEq("")));
}
#else
TEST(RustValidatorParity, SkipWithoutRustTarget) { GTEST_SKIP(); }
#endif  // SPIRV_RUST_TARGET_ENV

}  // namespace
}  // namespace spvtools
