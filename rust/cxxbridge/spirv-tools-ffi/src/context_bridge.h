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

#pragma once

#include <cstddef>
#include <cstdint>

namespace rust {
inline namespace cxxbridge1 {
class Str;
template <typename T>
class Slice;
}  // namespace cxxbridge1
}  // namespace rust

namespace spvtools::ffi {

struct MessagePosition;
struct AssembleResult;
struct ValidateResult;
struct ReduceResult;
struct FuzzResult;

void dispatch_context_message(std::size_t context_ptr, std::uint32_t level,
                              bool has_source, rust::Str source,
                              MessagePosition position, rust::Str message);

AssembleResult assemble_text_with_context(std::size_t context_ptr,
                                          rust::Slice<const std::uint8_t> text,
                                          std::uint32_t options);

ValidateResult validate_binary(std::uint32_t env,
                               rust::Slice<const std::uint32_t> words);

ReduceResult reduce_with_cpp(std::uint32_t env,
                             rust::Slice<const std::uint32_t> words);

FuzzResult fuzz_with_cpp(std::uint32_t env,
                         rust::Slice<const std::uint32_t> words);

}  // namespace spvtools::ffi
