// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

#include "source/spirv_target_env.h"
#include "source/spirv_validator_options.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/io.h"
#include "tools/util/cli_consumer.h"
#if defined(SPIRV_RUST_TARGET_ENV)
#include "rust/cxxbridge/spirv-tools-ffi.h"
#else
namespace spvtools {
namespace ffi {
struct ValidatorOptions {};
}  // namespace ffi
}  // namespace spvtools
#endif

void print_usage(char* argv0) {
  std::string target_env_list = spvTargetEnvList(36, 105);
  printf(
      R"(%s - Validate a SPIR-V binary file(s).

USAGE: %s [options] [<path>]

The SPIR-V binary is read from <path>. If no path is specified,
or if the path is "-", then the binary is read from standard input.
The <path> parameter may also specify a directory; in this case,
the tool will recursively process all regular files with the .spv
extension within that directory.

NOTE: The validator is a work in progress.

Options:
  -h, --help                       Print this help.
  --max-struct-members             <maximum number of structure members allowed>
  --max-struct-depth               <maximum allowed nesting depth of structures>
  --max-local-variables            <maximum number of local variables allowed>
  --max-global-variables           <maximum number of global variables allowed>
  --max-switch-branches            <maximum number of branches allowed in switch statements>
  --max-function-args              <maximum number arguments allowed per function>
  --max-control-flow-nesting-depth <maximum Control Flow nesting depth allowed>
  --max-access-chain-indexes       <maximum number of indexes allowed to use for Access Chain instructions>
  --max-id-bound                   <maximum value for the id bound>
  --relax-logical-pointer          Allow allocating an object of a pointer type and returning
                                   a pointer value from a function in logical addressing mode
  --relax-block-layout             Enable VK_KHR_relaxed_block_layout when checking standard
                                   uniform, storage buffer, and push constant layouts.
                                   This is the default when targeting Vulkan 1.1 or later.
  --uniform-buffer-standard-layout Enable VK_KHR_uniform_buffer_standard_layout when checking standard
                                   uniform buffer layouts.
  --scalar-block-layout            Enable VK_EXT_scalar_block_layout when checking standard
                                   uniform, storage buffer, and push constant layouts.  Scalar layout
                                   rules are more permissive than relaxed block layout so in effect
                                   this will override the --relax-block-layout option.
  --workgroup-scalar-block-layout  Enable scalar block layout when checking Workgroup block layouts.
  --skip-block-layout              Skip checking standard uniform/storage buffer layout.
                                   Overrides any --relax-block-layout or --scalar-block-layout option.
  --relax-struct-store             Allow store from one struct type to a
                                   different type with compatible layout and
                                   members.
  --allow-localsizeid              Allow use of the LocalSizeId decoration where it would otherwise not
                                   be allowed by the target environment.
  --allow-offset-texture-operand   Allow use of the Offset texture operands where it would otherwise not
                                   be allowed by the target environment.
  --allow-vulkan-32-bit-bitwise    Allow use of non-32 bit for the Base operand where it would otherwise
                                   not be allowed by the target environment.
  --before-hlsl-legalization       Allows code patterns that are intended to be
                                   fixed by spirv-opt's legalization passes.
#if defined(SPIRV_RUST_TARGET_ENV)
  --prefer-rust-validator          Default to the Rust validator when both are available.
  --prefer-cpp-validator           Default to the legacy C++ validator when both are available.
  --force-rust-validator           Prefer the Rust validator even when both are available.
  --force-cpp-validator            Force the legacy C++ validator (disables the Rust path).
#endif
  --version                        Display validator version information.
  --target-env                     {%s}
                                    Use validation rules from the specified environment.
)",
      argv0, argv0, target_env_list.c_str());
}

bool process_single_file(const char* filename, spv_target_env& target_env,
                         spvtools::ValidatorOptions& options,
                         bool use_default_msg_consumer,
                         bool use_rust_validator,
                         const spvtools::ffi::ValidatorOptions& rust_options) {
#if !defined(SPIRV_RUST_TARGET_ENV)
  (void)use_rust_validator;
  (void)rust_options;
#endif
  std::vector<uint32_t> contents;
  if (!ReadBinaryFile(filename, &contents)) return false;

#if defined(SPIRV_RUST_TARGET_ENV)
  if (use_rust_validator) {
    const auto result = spvtools::ffi::validate_binary_with_options(
        static_cast<uint32_t>(target_env), contents, rust_options);
    if (result.success) return true;

    const char* pretty_filename = filename ? filename : "stdin";
    std::cerr << "error: " << pretty_filename << ": " << result.message
              << std::endl;
    return false;
  }
#endif

  spvtools::SpirvTools tools(target_env);

  // Use a lambda expression here so filename can be captured. Messages use a
  // fairly standard notation of `filename:line`.
  auto CLIMessageConsumerWithFilename =
      [filename](spv_message_level_t level, const char*,
                 const spv_position_t& position, const char* message) {
        const char* pretty_filename = filename;
        if (!filename || 0 == strcmp(filename, "-")) {
          pretty_filename = "stdin";
        }

        switch (level) {
          case SPV_MSG_FATAL:
          case SPV_MSG_INTERNAL_ERROR:
          case SPV_MSG_ERROR:
            std::cerr << "error: " << pretty_filename << ":" << position.index
                      << ": " << message << std::endl;
            break;
          case SPV_MSG_WARNING:
            std::cout << "warning: " << pretty_filename << ":" << position.index
                      << ": " << message << std::endl;
            break;
          case SPV_MSG_INFO:
            std::cout << "info: " << pretty_filename << ":" << position.index
                      << ": " << message << std::endl;
            break;
          default:
            break;
        }
      };

  if (use_default_msg_consumer) {
    tools.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);
  } else {
    tools.SetMessageConsumer(CLIMessageConsumerWithFilename);
  }

  return tools.Validate(contents.data(), contents.size(), options);
}

int main(int argc, char** argv) {
  const char* inFile = nullptr;
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;
  spvtools::ValidatorOptions options;
  bool continue_processing = true;
  int return_code = 0;
  spvtools::ffi::ValidatorOptions rust_options = {};
#if defined(SPIRV_RUST_TARGET_ENV)
  rust_options.use_friendly_names = true;
#endif
#if defined(SPIRV_RUST_TARGET_ENV)
  const bool env_disables_rust =
      std::getenv("SPIRV_TOOLS_DISABLE_RUST_VALIDATOR") != nullptr;
  const bool env_forces_rust =
      !env_disables_rust &&
      std::getenv("SPIRV_TOOLS_FORCE_RUST_VALIDATOR") != nullptr;
  const bool env_prefers_cpp =
      !env_disables_rust &&
      std::getenv("SPIRV_TOOLS_PREFER_CPP_VALIDATOR") != nullptr;
  const bool env_prefers_rust =
      !env_disables_rust &&
      std::getenv("SPIRV_TOOLS_PREFER_RUST_VALIDATOR") != nullptr;
  bool prefer_rust_validator = !env_disables_rust;
  bool prefer_cpp_validator = false;
  bool force_rust_validator = env_forces_rust;
  bool force_cpp_validator = false;
  if (env_prefers_cpp) {
    prefer_cpp_validator = true;
    prefer_rust_validator = false;
  } else if (env_prefers_rust) {
    prefer_rust_validator = true;
    prefer_cpp_validator = false;
  }
#endif

  for (int argi = 1; continue_processing && argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strncmp(cur_arg, "--max-", 6)) {
        if (argi + 1 < argc) {
          spv_validator_limit limit_type;
          if (spvParseUniversalLimitsOptions(cur_arg, &limit_type)) {
            uint32_t limit = 0;
            if (sscanf(argv[++argi], "%u", &limit)) {
              options.SetUniversalLimit(limit_type, limit);
#if defined(SPIRV_RUST_TARGET_ENV)
              rust_options.limits.push_back(
                  spvtools::ffi::ValidatorLimit{static_cast<uint32_t>(limit_type), limit});
#endif
            } else {
              fprintf(stderr, "error: missing argument to %s\n", cur_arg);
              continue_processing = false;
              return_code = 1;
            }
          } else {
            fprintf(stderr, "error: unrecognized option: %s\n", cur_arg);
            continue_processing = false;
            return_code = 1;
          }
        } else {
          fprintf(stderr, "error: Missing argument to %s\n", cur_arg);
          continue_processing = false;
          return_code = 1;
        }
      } else if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\n", spvSoftwareVersionDetailsString());
        printf(
            "Targets:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  "
            "%s\n  %s\n  %s\n  %s %s\n",
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_0),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_1),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_2),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_3),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_4),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_5),
            spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_6),
            spvTargetEnvDescription(SPV_ENV_OPENCL_2_2),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_0),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_1),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_1_SPIRV_1_4),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_2),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_3),
            spvTargetEnvDescription(SPV_ENV_VULKAN_1_4));
        continue_processing = false;
        return_code = 0;
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        print_usage(argv[0]);
        continue_processing = false;
        return_code = 0;
      } else if (0 == strcmp(cur_arg, "--target-env")) {
        if (argi + 1 < argc) {
          const auto env_str = argv[++argi];
          if (!spvParseTargetEnv(env_str, &target_env)) {
            fprintf(stderr, "error: Unrecognized target env: %s\n", env_str);
            continue_processing = false;
            return_code = 1;
          }
        } else {
          fprintf(stderr, "error: Missing argument to --target-env\n");
          continue_processing = false;
          return_code = 1;
        }
      } else if (0 == strcmp(cur_arg, "--before-hlsl-legalization")) {
        options.SetBeforeHlslLegalization(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.before_hlsl_legalization = true;
        rust_options.relax_logical_pointer = true;
#endif
      } else if (0 == strcmp(cur_arg, "--relax-logical-pointer")) {
        options.SetRelaxLogicalPointer(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.relax_logical_pointer = true;
#endif
      } else if (0 == strcmp(cur_arg, "--relax-block-layout")) {
        options.SetRelaxBlockLayout(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.relax_block_layout = true;
#endif
      } else if (0 == strcmp(cur_arg, "--uniform-buffer-standard-layout")) {
        options.SetUniformBufferStandardLayout(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.uniform_buffer_standard_layout = true;
#endif
      } else if (0 == strcmp(cur_arg, "--scalar-block-layout")) {
        options.SetScalarBlockLayout(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.scalar_block_layout = true;
#endif
      } else if (0 == strcmp(cur_arg, "--workgroup-scalar-block-layout")) {
        options.SetWorkgroupScalarBlockLayout(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.workgroup_scalar_block_layout = true;
#endif
      } else if (0 == strcmp(cur_arg, "--skip-block-layout")) {
        options.SetSkipBlockLayout(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.skip_block_layout = true;
#endif
      } else if (0 == strcmp(cur_arg, "--allow-localsizeid")) {
        options.SetAllowLocalSizeId(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.allow_localsizeid = true;
#endif
      } else if (0 == strcmp(cur_arg, "--allow-offset-texture-operand")) {
        options.SetAllowOffsetTextureOperand(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.allow_offset_texture_operand = true;
#endif
      } else if (0 == strcmp(cur_arg, "--allow-vulkan-32-bit-bitwise")) {
        options.SetAllowVulkan32BitBitwise(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.allow_vulkan_32_bit_bitwise = true;
#endif
      } else if (0 == strcmp(cur_arg, "--relax-struct-store")) {
        options.SetRelaxStructStore(true);
#if defined(SPIRV_RUST_TARGET_ENV)
        rust_options.relax_struct_store = true;
#endif
#if defined(SPIRV_RUST_TARGET_ENV)
      } else if (0 == strcmp(cur_arg, "--force-rust-validator")) {
        force_rust_validator = true;
      } else if (0 == strcmp(cur_arg, "--force-cpp-validator")) {
        force_cpp_validator = true;
      } else if (0 == strcmp(cur_arg, "--prefer-rust-validator")) {
        prefer_rust_validator = true;
        prefer_cpp_validator = false;
      } else if (0 == strcmp(cur_arg, "--prefer-cpp-validator")) {
        prefer_cpp_validator = true;
        prefer_rust_validator = false;
#endif
      } else if (0 == cur_arg[1]) {
        // Setting a filename of "-" to indicate stdin.
        if (!inFile) {
          inFile = cur_arg;
        } else {
          fprintf(stderr, "error: More than one input file specified\n");
          continue_processing = false;
          return_code = 1;
        }
      } else {
        print_usage(argv[0]);
        continue_processing = false;
        return_code = 1;
      }
    } else {
      if (!inFile) {
        inFile = cur_arg;
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        continue_processing = false;
        return_code = 1;
      }
    }
  }

  // Exit if command line parsing was not successful.
  if (!continue_processing) {
    return return_code;
  }

#if defined(SPIRV_RUST_TARGET_ENV)
  bool use_rust_validator = prefer_rust_validator && !prefer_cpp_validator;
  if (force_cpp_validator) use_rust_validator = false;
  if (force_rust_validator) use_rust_validator = true;
#else
  bool use_rust_validator = false;
#endif

  if (inFile &&
      std::filesystem::is_directory(std::filesystem::status(inFile))) {
    const std::filesystem::path dir(inFile);
    bool succeed = true;
    for (auto const& entry :
         std::filesystem::recursive_directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      std::filesystem::path filepath = entry.path();

      if (filepath.extension() != ".spv") continue;

      // Copy the string, because in C++20 the result type of
      // std::filesystem::path::u8string changes type from std::string to
      // std::u8string, and the pointer type ends up incompatible. Normalize
      // to std::string first via copying.
      const auto filepath_u8str = filepath.u8string();
      const std::string filepath_str(filepath_u8str.begin(),
                                     filepath_u8str.end());
      if (!process_single_file(filepath_str.c_str(), target_env, options,
                               false, use_rust_validator, rust_options)) {
        succeed = false;
      }
    }

    return !succeed;
  }

  return !process_single_file(inFile, target_env, options, true,
                              use_rust_validator, rust_options);
}
