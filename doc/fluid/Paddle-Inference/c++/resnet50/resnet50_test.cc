#include <numeric>
#include <iostream>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle::AnalysisConfig;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");

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
  }
  config.SetModel(FLAGS_model_file,
                     FLAGS_params_file);
  config.EnableUseGpu(100, 0);
  // We use ZeroCopy, so we set config->SwitchUseFeedFetchOps(false)
  config.SwitchUseFeedFetchOps(false);
  config.EnableMKLDNN();

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePaddlePredictor(config);
}

void run(paddle::PaddlePredictor *predictor,
         const std::vector<float>& input,
         const std::vector<int>& input_shape, 
         std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->copy_from_cpu(input.data());

  CHECK(predictor->ZeroCopyRun());

  auto output_names = predictor->GetOutputNames();
  // there is only one output of Resnet50
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data->resize(out_num);
  output_t->copy_to_cpu(out_data->data());
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = CreatePredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224}; 
  // init 0 for the input. 
  std::vector<float> input_data(FLAGS_batch_size * 3 * 224 * 224, 0); 
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);
  
  for (auto e : out_data) {
    LOG(INFO) << e << std::endl;
  }
  return 0;
}
