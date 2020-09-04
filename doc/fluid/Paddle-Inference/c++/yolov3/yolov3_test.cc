#include "paddle/include/paddle_inference_api.h"

#include <numeric>
#include <iostream>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>

using paddle::AnalysisConfig;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "enable gpu");
DEFINE_bool(use_mkldnn, false, "enable mkldnn");
DEFINE_bool(mem_optim, false, "enable memory optimize");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor() {
  AnalysisConfig config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  } else {
    config.SetModel(FLAGS_model_file,
                    FLAGS_params_file);
  }
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  if (FLAGS_use_mkldnn) {
    config.EnableMKLDNN();
  }
  // Open the memory optim.
  if (FLAGS_mem_optim) {
    config.EnableMemoryOptim();
  }
  // We use ZeroCopy, so we set config->SwitchUseFeedFetchOps(false)
  config.SwitchUseFeedFetchOps(false);
  return CreatePaddlePredictor(config);
}

void run(paddle::PaddlePredictor *predictor,
         const std::vector<float>& input,
         const std::vector<int>& input_shape, 
         const std::vector<int>& input_im,
         const std::vector<int>& input_im_shape,
         std::vector<float> *out_data) {
  auto input_names = predictor->GetInputNames();
  auto input_img = predictor->GetInputTensor(input_names[0]);
  input_img->Reshape(input_shape);
  input_img->copy_from_cpu(input.data());

  auto input_size = predictor->GetInputTensor(input_names[1]);
  input_size->Reshape(input_im_shape);
  input_size->copy_from_cpu(input_im.data());

  CHECK(predictor->ZeroCopyRun());

  auto output_names = predictor->GetOutputNames();
  // there is only one output of yolov3
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data->resize(out_num);
  output_t->copy_to_cpu(out_data->data());
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = CreatePredictor();

  const int height = 608;
  const int width = 608;
  const int channels = 3;
  std::vector<int> input_shape = {FLAGS_batch_size, channels, height, width};
  std::vector<float> input_data(FLAGS_batch_size * channels * height * width, 0);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = i % 255 * 0.13f;
  }
  std::vector<int> input_im_shape = {FLAGS_batch_size, 2};
  std::vector<int> input_im_data(FLAGS_batch_size * 2, 608);

  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, input_im_data, input_im_shape, &out_data);
  LOG(INFO) << "output num is " << out_data.size();
  return 0;
}
