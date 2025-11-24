#include <cstdint>
#include <sstream>
#include <string>

#include "rust/cxxbridge/spirv-tools-ffi/src/context_bridge.h"
#include "rust/cxxbridge/spirv-tools-ffi.h"
#include "source/table.h"
#include "spirv-tools/libspirv.hpp"

namespace {
std::string FormatDiagnostic(spv_message_level_t, const spv_position_t& position,
                             const char* message) {
  std::ostringstream oss;
  oss << "error";
  if (position.line) {
    oss << " (line " << position.line << ", column " << position.column << ")";
  }
  oss << ": " << message;
  return oss.str();
}
}  // namespace

namespace spvtools::ffi {
namespace {
spv_position_t ToSpvPosition(MessagePosition position) {
  spv_position_t pos = {};
  pos.line = position.line;
  pos.column = position.column;
  pos.index = position.index;
  return pos;
}
}  // namespace

void dispatch_context_message(std::uintptr_t context_ptr, std::uint32_t level,
                              bool has_source, rust::Str source,
                              MessagePosition position, rust::Str message) {
  auto* context = reinterpret_cast<spv_context>(context_ptr);
  if (context == nullptr || !context->consumer) {
    return;
  }

  std::string message_storage(message.data(), message.length());
  const char* source_ptr = nullptr;
  std::string source_storage;
  if (has_source) {
    source_storage.assign(source.data(), source.length());
    source_ptr = source_storage.c_str();
  }

  context->consumer(static_cast<spv_message_level_t>(level), source_ptr,
                    ToSpvPosition(position), message_storage.c_str());
}

ValidateResult validate_binary(std::uint32_t env,
                               rust::Slice<const std::uint32_t> words) {
  ValidateResult result{false, ::rust::String()};
  const spv_target_env target = static_cast<spv_target_env>(env);
  spvtools::SpirvTools tools(target);
  std::string diagnostics;
  tools.SetMessageConsumer([&diagnostics](spv_message_level_t level, const char*,
                                         const spv_position_t& position,
                                         const char* message) {
    if (!diagnostics.empty()) diagnostics += '\n';
    diagnostics += FormatDiagnostic(level, position, message);
  });
  if (tools.Validate(words.data(), words.size())) {
    result.success = true;
  } else {
    result.message = ::rust::String(diagnostics);
  }
  return result;
}

}  // namespace spvtools::ffi
