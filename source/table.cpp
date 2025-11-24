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

#include "source/table.h"

#include <cstddef>
#include <utility>

#if defined(SPIRV_RUST_TARGET_ENV)
#include "rust/cxxbridge/spirv-tools-ffi.h"
#endif

spv_context spvContextCreate(spv_target_env env) {
#if defined(SPIRV_RUST_TARGET_ENV)
  if (!spvtools::ffi::is_valid_env(static_cast<uint32_t>(env))) {
    return nullptr;
  }
#else
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
    case SPV_ENV_VULKAN_1_4:
      break;
    default:
      return nullptr;
  }
#endif

  auto* context = new spv_context_t{env, nullptr /* a null default consumer */
#if defined(SPIRV_RUST_TARGET_ENV)
                                    , nullptr
#endif
  };

#if defined(SPIRV_RUST_TARGET_ENV)
  auto rust_context = spvtools::ffi::create_context(
      static_cast<uint32_t>(env),
      reinterpret_cast<std::size_t>(context));
  if (rust_context == 0) {
    delete context;
    return nullptr;
  }
  context->rust_context = reinterpret_cast<void*>(rust_context);
#endif

  return context;
}

void spvContextDestroy(spv_context context) {
#if defined(SPIRV_RUST_TARGET_ENV)
  if (context && context->rust_context) {
    spvtools::ffi::destroy_context(
        reinterpret_cast<std::size_t>(context->rust_context));
    context->rust_context = nullptr;
  }
#endif
  delete context;
}

void spvtools::SetContextMessageConsumer(spv_context context,
                                         spvtools::MessageConsumer consumer) {
  context->consumer = std::move(consumer);
}

#if defined(SPIRV_RUST_TARGET_ENV)
uint64_t spvtools::GetRustContextHandle(spv_const_context context) {
  if (!context || !context->rust_context) {
    return 0;
  }
  const auto raw = reinterpret_cast<std::size_t>(context->rust_context);
  return spvtools::ffi::context_handle_from_raw(raw);
}

spvtools::ScopedRebindRustContext::ScopedRebindRustContext(
    uint64_t handle, spv_const_context original, spv_context_t* replacement)
    : handle_(handle),
      original_(reinterpret_cast<std::size_t>(original)),
      bound_(false) {
  if (handle_ != 0 && replacement != nullptr) {
    spvtools::ffi::rebind_context(
        handle_, reinterpret_cast<std::size_t>(replacement));
    bound_ = true;
  }
}

spvtools::ScopedRebindRustContext::~ScopedRebindRustContext() {
  if (bound_ && original_ != 0) {
    spvtools::ffi::rebind_context(handle_, original_);
  }
}
#endif
