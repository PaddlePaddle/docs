#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include "paddle/include/paddle_inference_api.h"

using paddle::AnalysisConfig;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(seq_len,
             128,
             "sequence length, should less than or equal to 512.");
DEFINE_bool(use_gpu, true, "enable gpu");

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
    config.SetModel(FLAGS_model_file, FLAGS_params_file);
  }
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  // We use ZeroCopy, so we set config->SwitchUseFeedFetchOps(false)
  config.SwitchUseFeedFetchOps(false);
  return CreatePaddlePredictor(config);
}

template <typename Dtype>
std::vector<Dtype> PrepareInput(const std::vector<int>& shape,
                                int word_size = 18000);

template <>
std::vector<float> PrepareInput(const std::vector<int>& shape, int word_size) {
  int count =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> datas(count, 0);
  for (int i = 0; i < count; ++i) {
    datas[i] = i % 2 ? 0.f : 1.f;
  }
  return datas;
}

template <>
std::vector<int64_t> PrepareInput(const std::vector<int>& shape,
                                  int word_size) {
  int count =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<int64_t> datas(count, 0);
  for (int i = 0; i < count; ++i) {
    datas[i] = (i + 13) % word_size == 0 ? 1 : (i + 13) % word_size;
  }
  return datas;
}

void Run(paddle::PaddlePredictor* predictor,
         std::vector<float>* out_data_0,
         std::vector<int64_t>* out_data_1,
         std::vector<int64_t>* out_data_2) {
  const int word_size_0 = 18000;
  const int word_size_1 = 2;
  const int word_size_2 = 513;

  const int batch_size = FLAGS_batch_size;
  const int seq_len = FLAGS_seq_len;

  auto input_names = predictor->GetInputNames();
  std::vector<int> shape_seq{batch_size, seq_len, 1};
  std::vector<int> shape_batch{batch_size};
  std::vector<int> shape_s{batch_size, seq_len};

#define INPUT_EMB(num)                                                 \
  auto input_##num = predictor->GetInputTensor(input_names[num]);      \
  auto data_##num = PrepareInput<int64_t>(shape_seq, word_size_##num); \
  input_##num->Reshape(shape_seq);                                     \
  input_##num->copy_from_cpu(data_##num.data())

  INPUT_EMB(0);
  INPUT_EMB(1);
  INPUT_EMB(2);

#undef INPUT_EMB

  auto input_3 = predictor->GetInputTensor(input_names[3]);
  auto data_3 = PrepareInput<float>(shape_seq);
  input_3->Reshape(shape_seq);
  input_3->copy_from_cpu(data_3.data());

  auto input_4 = predictor->GetInputTensor(input_names[4]);
  auto data_4 = PrepareInput<int64_t>(shape_batch);
  input_4->Reshape(shape_batch);
  input_4->copy_from_cpu(data_4.data());

#define INPUT_5_or_6(num)                                         \
  auto input_##num = predictor->GetInputTensor(input_names[num]); \
  auto data_##num = PrepareInput<int64_t>(shape_s, 1);            \
  input_##num->Reshape(shape_s);                                  \
  input_##num->copy_from_cpu(data_##num.data())

  INPUT_5_or_6(5);
  INPUT_5_or_6(6);

#undef INPUT_5_or_6

  CHECK(predictor->ZeroCopyRun());

  auto output_names = predictor->GetOutputNames();
  // there is three output of lic2020 baseline model

#define OUTPUT(num)                                                  \
  auto output_##num = predictor->GetOutputTensor(output_names[num]); \
  std::vector<int> output_shape_##num = output_##num->shape();       \
  int out_num_##num = std::accumulate(output_shape_##num.begin(),    \
                                      output_shape_##num.end(),      \
                                      1,                             \
                                      std::multiplies<int>());       \
  out_data_##num->resize(out_num_##num);                             \
  output_##num->copy_to_cpu(out_data_##num->data())

  OUTPUT(0);
  OUTPUT(1);
  OUTPUT(2);

#undef OUTPUT
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = CreatePredictor();

  std::vector<float> out_data_0;
  std::vector<int64_t> out_data_1;
  std::vector<int64_t> out_data_2;
  Run(predictor.get(), &out_data_0, &out_data_1, &out_data_2);

  LOG(INFO) << "output0 num is " << out_data_0.size();
  LOG(INFO) << "output1 num is " << out_data_1.size();
  LOG(INFO) << "output2 num is " << out_data_2.size();
  return 0;
}
