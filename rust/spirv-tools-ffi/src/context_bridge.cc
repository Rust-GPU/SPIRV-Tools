#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "spirv-tools-ffi/src/lib.rs.h"
#include "rust/cxxbridge/spirv-tools-ffi/src/context_bridge.h"
#include "spirv-tools/libspirv.h"

// When building with Rust target env, we don't need C++ fallback implementations
// since Rust provides them. This avoids link dependencies on C++ SPIRV-Tools
// libraries (libspirv.hpp, reducer.h) that would create circular dependencies.
#ifndef SPIRV_RUST_TARGET_ENV
#include "source/reduce/reducer.h"
#include "source/spirv_reducer_options.h"
#include "source/table.h"
#include "spirv-tools/libspirv.hpp"
#endif

namespace {
#ifndef SPIRV_RUST_TARGET_ENV
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
#endif
}  // namespace

namespace spvtools::ffi {

// In CMake builds with SPIRV_RUST_TARGET_ENV, dispatch_context_message and
// assemble_text_with_context are provided by source/text.cpp using internal APIs.
// In standalone Rust builds (Bazel), we provide them here.
#ifndef SPIRV_RUST_TARGET_ENV
namespace {
spv_position_t ToSpvPosition(MessagePosition position) {
  spv_position_t pos = {};
  pos.line = position.line;
  pos.column = position.column;
  pos.index = position.index;
  return pos;
}
}  // namespace

void dispatch_context_message(std::size_t context_ptr, std::uint32_t level,
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

AssembleResult assemble_text_with_context(std::size_t context_ptr,
                                          rust::Slice<const std::uint8_t> text,
                                          std::uint32_t options) {
  AssembleResult result{false, ::rust::Vec<std::uint32_t>()};
  auto* context = reinterpret_cast<spv_context>(context_ptr);
  if (context == nullptr) {
    return result;
  }

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  const char* text_ptr = reinterpret_cast<const char*>(text.data());
  const spv_result_t status =
      spvTextToBinaryWithOptions(context, text_ptr, text.size(), options,
                                 &binary, &diagnostic);
  if (diagnostic) {
    spvDiagnosticDestroy(diagnostic);
  }

  if (status == SPV_SUCCESS && binary != nullptr) {
    result.success = true;
    result.binary.reserve(binary->wordCount);
    for (size_t i = 0; i < binary->wordCount; ++i) {
      result.binary.push_back(binary->code[i]);
    }
    spvBinaryDestroy(binary);
  }

  return result;
}
#endif

ValidateResult validate_binary_with_options(
    std::uint32_t env, rust::Slice<const std::uint32_t> words,
    const ValidatorOptions& options) {
#ifdef SPIRV_RUST_TARGET_ENV
  // When built with Rust target env, always use Rust validator
  return validate_binary_rust(env, words, options);
#else
  ValidateResult result{false, ::rust::String()};
  if (rust_validator_enabled()) {
    return validate_binary_rust(env, words, options);
  }

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
#endif
}

ValidateResult validate_binary(std::uint32_t env,
                               rust::Slice<const std::uint32_t> words) {
  auto options = default_validator_options();
  return validate_binary_with_options(env, words, options);
}

ReduceResult reduce_with_cpp(std::uint32_t env,
                             rust::Slice<const std::uint32_t> words,
                             const ReduceOptions& options) {
#ifdef SPIRV_RUST_TARGET_ENV
  // When built with Rust target env, C++ reducer is not available.
  // The Rust side should handle reduction.
  (void)env;
  (void)words;
  (void)options;
  ReduceResult result{/*success=*/false,
                      ToolError::Disabled,
                      ::rust::String("C++ reducer unavailable in Rust build; use Rust reducer"),
                      ::rust::Vec<std::uint32_t>()};
  return result;
#else
  ReduceResult result{/*success=*/false,
                      ToolError::Parse,
                      ::rust::String(),
                      ::rust::Vec<std::uint32_t>()};
  if (words.empty()) {
    result.message = ::rust::String("empty module");
    return result;
  }

  spvtools::reduce::Reducer reducer(static_cast<spv_target_env>(env));
  std::string diagnostics;
  reducer.SetMessageConsumer(
      [&diagnostics](spv_message_level_t level, const char*,
                     const spv_position_t& position, const char* message) {
        if (!diagnostics.empty()) diagnostics += '\n';
        diagnostics += FormatDiagnostic(level, position, message);
      });

  // Keep behavior deterministic until the Rust reducer path lands: use no
  // reduction passes and a trivially-true interestingness predicate so the
  // original module is preserved while exercising the C++ reducer plumbing.
  reducer.SetInterestingnessFunction(
      [](const std::vector<std::uint32_t>&, std::uint32_t) { return true; });

  spv_validator_options validator_options = spvValidatorOptionsCreate();
  std::vector<std::uint32_t> input(words.begin(), words.end());
  std::vector<std::uint32_t> reduced;
  spv_reducer_options reducer_options = spvReducerOptionsCreate();
  spvReducerOptionsSetStepLimit(reducer_options, options.step_limit);
  spvReducerOptionsSetFailOnValidationError(reducer_options,
                                            options.fail_on_validation_error);
  spvReducerOptionsSetTargetFunction(reducer_options, options.target_function);
  const auto status =
      reducer.Run(input, &reduced, reducer_options, validator_options);
  spvReducerOptionsDestroy(reducer_options);
  spvValidatorOptionsDestroy(validator_options);

  switch (status) {
    case spvtools::reduce::Reducer::ReductionResultStatus::kComplete:
    case spvtools::reduce::Reducer::ReductionResultStatus::kReachedStepLimit:
    case spvtools::reduce::Reducer::ReductionResultStatus::kInitialStateNotInteresting:
      result.success = true;
      result.error = ToolError::None;
      result.message = ::rust::String();
      {
        const auto& chosen = reduced.empty() ? input : reduced;
        result.words.reserve(chosen.size());
        for (auto value : chosen) {
          result.words.push_back(value);
        }
      }
      break;
    case spvtools::reduce::Reducer::ReductionResultStatus::kInitialStateInvalid:
    case spvtools::reduce::Reducer::ReductionResultStatus::kStateInvalid:
      result.success = false;
      result.error = ToolError::Parse;
      if (diagnostics.empty()) {
        result.message = ::rust::String("reducer validation failed");
      } else {
        result.message = ::rust::String(diagnostics);
      }
      break;
  }

  return result;
#endif
}

FuzzResult fuzz_with_cpp(std::uint32_t env,
                         rust::Slice<const std::uint32_t> words,
                         const FuzzOptions&) {
  (void)env;
  (void)words;
  FuzzResult result{/*success=*/false,
                    ToolError::Disabled,
                    ::rust::String("C++ fuzz bridge unavailable; use Rust fuzz pipeline"),
                    ::rust::Vec<std::uint32_t>()};
  return result;
}

}  // namespace spvtools::ffi
