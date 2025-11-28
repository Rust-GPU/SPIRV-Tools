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

#include "gmock/gmock.h"
#include "rust/cxxbridge/spirv-tools-ffi.h"
#include "source/spirv_constant.h"

namespace spvtools {
namespace {

using ::testing::Not;
using ::testing::StrEq;

TEST(RustDisassemblerDiagnostics, SurfacesParseErrorsThroughFFI) {
  const uint32_t truncated_binary[] = {spv::MagicNumber};
  ::rust::Slice<const uint32_t> binary(truncated_binary,
                                       sizeof(truncated_binary) /
                                           sizeof(truncated_binary[0]));

  // Context handle is ignored for diagnostic capture; zero is sufficient.
  auto result = ffi::try_disassemble_binary(
      /*context_handle=*/0, binary, SPV_BINARY_TO_TEXT_OPTION_NONE);
  EXPECT_FALSE(result.success);
  ASSERT_FALSE(result.diagnostics.empty());

  const auto& diagnostic = result.diagnostics.front();
  EXPECT_TRUE(diagnostic.has_source);
  EXPECT_THAT(static_cast<std::string>(diagnostic.source),
              StrEq("disassembler"));
  EXPECT_THAT(static_cast<std::string>(diagnostic.message), Not(StrEq("")));
}

}  // namespace
}  // namespace spvtools
