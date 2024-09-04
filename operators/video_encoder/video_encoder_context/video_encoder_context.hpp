/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_CONTEXT_VIDEO_ENCODER_CONTEXT
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_CONTEXT_VIDEO_ENCODER_CONTEXT

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

namespace holoscan::ops {

/**
 * @brief Encoder context class shared by `VideoEncoderRequestOp` and
 * `VideoEncoderResponseOp`.
 *
 * This wraps a GXF Component(`nvidia::gxf::VideoEncoderContext`).
 */
class VideoEncoderContext: public holoscan::gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(VideoEncoderContext,
      holoscan::gxf::GXFResource)

  VideoEncoderContext() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoEncoderContext";
  };

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<std::shared_ptr<holoscan::AsynchronousCondition>> scheduling_term_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_ENCODER_CONTEXT_VIDEO_ENCODER_CONTEXT
