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
#include <string>

#include "gmock/gmock.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StrEq;

#if defined(SPIRV_RUST_TARGET_ENV)
TEST(RustDisassemblerFallback, ForwardsRustDiagnosticsToMessageConsumer) {
  // Force the Rust disassembler path on so diagnostics are produced there first.
  setenv("SPIRV_RUST_USE_DISASSEMBLER", "1", /*overwrite=*/1);
  auto* context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  ASSERT_NE(context, nullptr);

  std::string captured;
  SetContextMessageConsumer(
      context, [&captured](spv_message_level_t, const char* source,
                           const spv_position_t&, const char* message) {
        if (source) {
          captured.append(source);
          captured.push_back(':');
        }
        captured.append(message);
      });

  // Truncated binary should fail in both Rust and C++ paths.
  const uint32_t truncated[] = {spv::MagicNumber};
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;
  const auto status = spvBinaryToText(
      context, truncated, sizeof(truncated) / sizeof(truncated[0]),
      SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic);
  EXPECT_EQ(status, SPV_ERROR_INVALID_BINARY);
  if (text) spvTextDestroy(text);
  if (diagnostic) spvDiagnosticDestroy(diagnostic);

  EXPECT_THAT(captured, Not(StrEq("")));
  EXPECT_THAT(captured, HasSubstr("disassembler"));

  spvContextDestroy(context);
  unsetenv("SPIRV_RUST_USE_DISASSEMBLER");
}
#endif  // defined(SPIRV_RUST_TARGET_ENV)

}  // namespace
}  // namespace spvtools
