/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <getopt.h>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/aja_source/aja_source.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#ifdef VTK_RENDERER
#include <vtk_renderer.hpp>
#endif

#include "holoscan/core/resources/gxf/gxf_component_resource.hpp"
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

#include "tensor_to_video_buffer.hpp"
#include "video_encoder.hpp"

#ifdef DELTACAST_VIDEOMASTER
#include <videomaster_source.hpp>
#include <videomaster_transmitter.hpp>
#endif

#ifdef YUAN_QCAP
#include <qcap_source.hpp>
#endif

// Import h.264 GXF codelets and components as Holoscan operators and resources
// Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
// resources can be done using the HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR and
// HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE macros. This new feature allows using GXF codelets
// and components in Holoscan applications without writing custom class wrappers (for C++) and
// Python wrappers (for Python) for each GXF codelet and component.
// For the VideoEncoderRequestOp class, since it needs to override the setup() to provide custom
// parameters and override the initialize() to register custom converters, it requires a custom
// class that extends the holoscan::ops::GXFCodeletOp class.

// The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
// of the decoded H264 bit stream.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder context
//   Handle.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderResponseOp, "nvidia::gxf::VideoDecoderResponse")

// The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
// for the H264 bit stream decode.
// Parameters:
// - inbuf_storage_type (uint32_t): Input Buffer storage type, 0:kHost, 1:kDevice.
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder
//   context Handle.
// - codec (uint32_t): Video codec to use, 0:H264, only H264 supported. Default:0.
// - disableDPB (uint32_t): Enable low latency decode, works only for IPPP case.
// - output_format (std::string): VidOutput frame video format, nv12pl and yuv420planar are
//   supported.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderRequestOp, "nvidia::gxf::VideoDecoderRequest")

// The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoDecoderContext, "nvidia::gxf::VideoDecoderContext")

// The VideoReadBitstreamOp implements nvidia::gxf::VideoReadBitStream and reads h.264 video files
// from the disk at the specified input file path.
// Parameters:
// - input_file_path (std::string): Path to image file
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data
// - outbuf_storage_type (int32_t): Output Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoReadBitstreamOp, "nvidia::gxf::VideoReadBitStream")

// The VideoWriteBitstreamOp implements nvidia::gxf::VideoWriteBitstream and writes bit stream to
// the disk at specified output path.
// Parameters:
// - output_video_path (std::string): The file path of the output video
// - frame_width (int): The width of the output video
// - frame_height (int): The height of the output video
// - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoWriteBitstreamOp, "nvidia::gxf::VideoWriteBitstream")

// The VideoEncoderResponseOp implements nvidia::gxf::VideoEncoderResponse and handles the output
// of the encoded YUV frames.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - videoencoder_context (std::shared_ptr<holoscan::ops::VideoEncoderContext>): Encoder context
//   handle.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice. Default: 1.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoEncoderResponseOp, "nvidia::gxf::VideoEncoderResponse")

// The VideoEncoderContext implements nvidia::gxf::VideoEncoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoEncoderContext, "nvidia::gxf::VideoEncoderContext")



class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) { source_ = source; }
  void set_visualizer_name(const std::string& visualizer_name) {
    this->visualizer_name = visualizer_name;
  }

  enum class Record { NONE, INPUT, VISUALIZER };

  void set_record(const std::string& record) {
    if (record == "input") {
      record_type_ = Record::INPUT;
    } else if (record == "visualizer") {
      record_type_ = Record::VISUALIZER;
    }
  }

  void set_datapath(const std::string& path) { datapath = path; }

  /// @brief As of Holoscan SDK 2.1.0, the extension manager must be used to register any external
  /// GXF extensions in replace of the use of YAML configuration file.
  void configure_extension() {
    auto extension_manager = executor().extension_manager();
    extension_manager->load_extension("libgxf_videodecoder.so");
    extension_manager->load_extension("libgxf_videodecoderio.so");
    extension_manager->load_extension("libgxf_videoencoder.so");
    extension_manager->load_extension("libgxf_videoencoderio.so");
  }

  void compose() override {
    using namespace holoscan;

    configure_extension();

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;
    std::shared_ptr<Operator> visualizer_operator;

    const bool use_rdma = from_config("external_source.rdma").as<bool>();
    const bool overlay_enabled = (source_ != "replayer") && (this->visualizer_name == "holoviz") &&
                                 from_config("external_source.enable_overlay").as<bool>();

    const std::string input_video_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "videostream";
    const std::string input_annotations_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "annotations";

    const bool record_output = from_config("record_output").as<bool>();

    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    if (source_ == "aja") {
      width = from_config("aja.width").as<uint32_t>();
      height = from_config("aja.height").as<uint32_t>();
      source = make_operator<ops::AJASourceOp>(
          "aja", from_config("aja"), from_config("external_source"));
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else if (source_ == "yuan") {
      width = from_config("yuan.width").as<uint32_t>();
      height = from_config("yuan.height").as<uint32_t>();
#ifdef YUAN_QCAP
      source = make_operator<ops::QCAPSourceOp>("yuan", from_config("yuan"));
#endif
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else if (source_ == "deltacast") {
      width = from_config("deltacast.width").as<uint32_t>();
      height = from_config("deltacast.height").as<uint32_t>();
#ifdef DELTACAST_VIDEOMASTER
      source = make_operator<ops::VideoMasterSourceOp>(
          "deltacast",
          from_config("deltacast"),
          from_config("external_source"),
          Arg("pool") = make_resource<UnboundedAllocator>("pool"));
#endif
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else {  // Replayer
      width = 854;
      height = 480;
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", datapath));
      source_block_size = width * height * 3 * 4;
      source_num_blocks = 2;
    }

    if (record_type_ != Record::NONE) {
      if (((record_type_ == Record::INPUT) && (source_ != "replayer")) ||
          (record_type_ == Record::VISUALIZER)) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      }
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter_" + source_),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = datapath + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath + "/engines";

    const uint64_t lstm_inferer_block_size = 107 * 60 * 7 * 4;
    const uint64_t lstm_inferer_num_blocks = 2 + 5 * 2;
    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, lstm_inferer_block_size, lstm_inferer_num_blocks),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") =
            make_resource<BlockMemoryPool>("device_allocator",
                                           1,
                                           tool_tracking_postprocessor_block_size,
                                           tool_tracking_postprocessor_num_blocks),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    if (this->visualizer_name == "holoviz") {
      std::shared_ptr<BlockMemoryPool> visualizer_allocator;
      if (((record_type_ == Record::VISUALIZER) && source_ == "replayer") || record_output ) {
        visualizer_allocator =
            make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
      }
      visualizer_operator = make_operator<ops::HolovizOp>(
          "holoviz",
          from_config(overlay_enabled ? "holoviz_overlay" : "holoviz"),
          Arg("width") = width,
          Arg("height") = height,
          Arg("enable_render_buffer_input") = overlay_enabled,
          Arg("enable_render_buffer_output") =
              overlay_enabled || (record_type_ == Record::VISUALIZER) || record_output,
          Arg("allocator") = visualizer_allocator,
          Arg("cuda_stream_pool") = cuda_stream_pool);

    }
#ifdef VTK_RENDERER
    if (this->visualizer_name == "vtk") {
      visualizer_operator = make_operator<ops::VtkRendererOp>("vtk",
                                                      from_config("vtk_op"),
                                                      Arg("width") = width,
                                                      Arg("height") = height);
    }
#endif

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});

    if (this->visualizer_name == "holoviz") {
      add_flow(tool_tracking_postprocessor,
               visualizer_operator,
               {{"out_coords", input_annotations_signal}, {"out_mask", input_annotations_signal}});
    } else {
      // device tensor on the out_mask port is not used by VtkRendererOp
      add_flow(tool_tracking_postprocessor,
               visualizer_operator,
               {{"out_coords", input_annotations_signal}});
    }

    std::string output_signal = "output";  // replayer output signal name
    if (source_ == "deltacast") {
      output_signal = "signal";
    } else if (source_ == "aja" || source_ == "yuan") {
      output_signal = "video_buffer_output";
    }

    add_flow(source, format_converter, {{output_signal, "source_video"}});

    add_flow(format_converter, lstm_inferer);

    if (source_ == "deltacast") {
#ifdef DELTACAST_VIDEOMASTER
      if (overlay_enabled) {
        // Overlay buffer flow between source and visualizer
        auto overlayer = make_operator<ops::VideoMasterTransmitterOp>(
            "videomaster_overlayer",
            from_config("videomaster"),
            Arg("pool") = make_resource<UnboundedAllocator>("pool"));
        auto overlay_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
            "overlay_format_converter",
            from_config("deltacast_overlay_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        add_flow(visualizer_operator,
                 overlay_format_converter_videomaster,
                 {{"render_buffer_output", ""}});
        add_flow(overlay_format_converter_videomaster, overlayer);
      } else {
        auto visualizer_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
            "visualizer_format_converter",
            from_config("deltacast_visualizer_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        auto drop_alpha_channel_converter = make_operator<ops::FormatConverterOp>(
            "drop_alpha_channel_converter",
            from_config("deltacast_drop_alpha_channel_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        add_flow(source, drop_alpha_channel_converter);
        add_flow(drop_alpha_channel_converter, visualizer_format_converter_videomaster);
        add_flow(visualizer_format_converter_videomaster, visualizer_operator, {{"", "receivers"}});
      }
#endif
    } else {
      if (overlay_enabled) {
        // Overlay buffer flow between source and visualizer_operator
        add_flow(source, visualizer_operator, {{"overlay_buffer_output", "render_buffer_input"}});
        add_flow(visualizer_operator, source, {{"render_buffer_output", "overlay_buffer_input"}});
      } else {
        add_flow(source, visualizer_operator, {{output_signal, input_video_signal}});
      }
    }

    if (record_type_ == Record::INPUT) {
      if (source_ != "replayer") {
        add_flow(source, recorder_format_converter, {{output_signal, "source_video"}});
        add_flow(recorder_format_converter, recorder);
      } else {
        add_flow(source, recorder);
      }
    } else if (record_type_ == Record::VISUALIZER && this->visualizer_name == "holoviz") {
      add_flow(visualizer_operator,
               recorder_format_converter,
               {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }

    if (record_output) {
      auto encoder_async_condition =
          make_condition<AsynchronousCondition>("encoder_async_condition");
      auto video_encoder_context =
          make_resource<VideoEncoderContext>(Arg("scheduling_term") = encoder_async_condition);

      auto video_encoder_request = make_operator<ops::VideoEncoderRequestOp>(
          "video_encoder_request",
          from_config("video_encoder_request"),
          Arg("videoencoder_context") = video_encoder_context);

      auto video_encoder_response = make_operator<VideoEncoderResponseOp>(
          "video_encoder_response",
          from_config("video_encoder_response"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
          Arg("videoencoder_context") = video_encoder_context);

      auto holoviz_output_format_converter = make_operator<ops::FormatConverterOp>(
          "holoviz_output_format_converter",
          from_config("holoviz_output_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

      auto encoder_input_format_converter = make_operator<ops::FormatConverterOp>(
          "encoder_input_format_converter",
          from_config("encoder_input_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

      auto tensor_to_video_buffer = make_operator<ops::TensorToVideoBufferOp>(
          "tensor_to_video_buffer", from_config("tensor_to_video_buffer"));

      auto bitstream_writer = make_operator<VideoWriteBitstreamOp>(
          "bitstream_writer",
          from_config("bitstream_writer"),
          Arg("output_video_path", datapath + "/surgical_video_output.264"),
          Arg("input_crc_file_path", datapath + "/surgical_video_output.txt"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 0, source_block_size, source_num_blocks));

      add_flow(
          visualizer_operator, holoviz_output_format_converter, {{"render_buffer_output", "source_video"}});
      add_flow(holoviz_output_format_converter,
               encoder_input_format_converter,
               {{"tensor", "source_video"}});
      add_flow(encoder_input_format_converter, tensor_to_video_buffer, {{"tensor", "in_tensor"}});
      add_flow(
          tensor_to_video_buffer, video_encoder_request, {{"out_video_buffer", "input_frame"}});
      add_flow(video_encoder_response, bitstream_writer, {{"output_transmitter", "data_receiver"}});
    }
  }

 private:
  std::string source_ = "replayer";
  std::string visualizer_name = "holoviz";
  Record record_type_ = Record::NONE;
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/endoscopy_tool_tracking.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);

  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  auto visualizer_name = app->from_config("visualizer").as<std::string>();
  app->set_visualizer_name(visualizer_name);

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
