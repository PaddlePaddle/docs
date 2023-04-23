# 算子数据类型扩展 验收规范

## 通过 CI 验证

提交至 Paddle repo 的 Pull Request（简称 PR），涉及到的相关检测 CI 必须全部 Pass。用来验证对之前功能点的兼容和影响，保障新合入代码对历史代码不产生影响。

新增代码必须要有相应的单测保障测试覆盖率达到准入要求（行覆盖率达到 90%）。

## 通过精度验证

扩展数据类型后需要添加对应数据类型的单元测试，并通过算子的精度检查。单元测试需要注意以下规范：
- [OP 单测必须使用大尺寸输入](https://github.com/PaddlePaddle/Paddle/wiki/OP-test-input-shape-requirements)
- [反向 Op 必须调用 check_grad](https://github.com/PaddlePaddle/Paddle/wiki/Gradient-Check-Is-Required-for-Op-Test)
- [单测精度中 atol, rtol, eps, max_relative_error, 不允许自行放大阈值](https://github.com/PaddlePaddle/Paddle/wiki/OP-test-accuracy-requirements)

## 通过性能验证

深度学习框架通常支持多种数据类型的输入，其中低精度运算不仅能够减少显存占用，还可以加快计算的效率。在一些特定硬件上，使用半精度浮点数 FP16 的峰值计算能力最高可达单精度浮点数 FP32 的数倍，基于此原理实现的混合精度训练策略对模型也可以实现数倍加速。在完成数据类型的扩展后，可以使用飞桨的[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)算子性能测试专业工具对算子性能进行测试对比，例如对于 FP16 数据类型，验收的基本要求是算子性能要明显优于使用 FP32 数据类型，同时我们也鼓励开发者针对 FP16 类型实现极致的加速。如下图所示，OP Benchmark 测试不同数据类型输入下的 OP 性能真实状态。

- Conv2d 算子，使用 FP32 数据类型：
```
===========================================================================
-- paddle version             : 0.0.0
-- paddle commit              : 5040e12e3ea3e640c14add6b9df70e9bfffb8271
-- benchmark commit           : 26a577ca0c92a9eedb5723dd8c5a57994f967f0e
-- benchmark last update time : Tue Jul 5 20:32:05 2022 +0800
===========================================================================
run command: nvprof --profile-from-start off /usr/bin/python /workspace/benchmark/api/dynamic_tests_v2/conv2d.py --task speed --framework paddle --testing_mode dynamic --json_file /workspace/benchmark/api/tests_v2/configs/conv2d.json --config_id 0 --profiler nvprof --backward False --use_gpu True --repeat 1000 --allow_adaptive_repeat False --log_level 0 --api_name paddle
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.86%  124.78ms      1000  124.78us  121.31us  128.77us  volta_sgemm_32x128_nn
                   31.03%  68.097ms      1000  68.096us  66.207us  77.024us  void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)
                    6.27%  13.767ms      1000  13.766us  12.384us  21.216us  void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)
                    5.84%  12.810ms      1000  12.810us  12.031us  18.463us  void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)

total gpu_time: 219.4513 ms

W0706 08:37:25.901571 20400 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.4, Runtime API Version: 11.4
W0706 08:37:25.901605 20400 gpu_context.cc:306] device: 0, cuDNN Version: 8.2.

[paddle][conv2d] conv2d {
  run_tf: True
  run_torch: True
  atol: 0.01
  data_format: NCHW
  dilation: [1, 1]
  groups: 1
  padding: [1, 1]
  stride: 1
  x_shape: [16, 512, 7, 7]
  x_dtype: float32
  weight_shape: [512, 512, 3, 3]
  weight_dtype: float32
}
```

- Conv2d 算子，使用 FP16 数据类型：
```
===========================================================================
-- paddle version             : 0.0.0
-- paddle commit              : 5040e12e3ea3e640c14add6b9df70e9bfffb8271
-- benchmark commit           : 26a577ca0c92a9eedb5723dd8c5a57994f967f0e
-- benchmark last update time : Tue Jul 5 20:32:05 2022 +0800
===========================================================================
run command: nvprof --profile-from-start off /usr/bin/python /workspace/benchmark/api/dynamic_tests_v2/conv2d.py --task speed --framework paddle --testing_mode dynamic --json_file /workspace/benchmark/api/tests_v2/configs/conv2d.json --config_id 0 --profiler nvprof --backward False --use_gpu True --repeat 1000 --allow_adaptive_repeat False --log_level 0 --convert_to_fp16 True --api_name paddle
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.33%  98.867ms      1000  98.867us  98.207us  106.46us  void xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm::fprop_indexed::Kernel_traits<xmma_cudnn::Volta_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Volta<int=0>, int=64, int=128, int=32, int=2, int=2, int=1, int=1>, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_a_t<xmma_cudnn::Volta_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Volta<int=0>, int=64, int=128, int=32, int=2, int=2, int=1, int=1>, xmma_cudnn::implicit_gemm::Input_related<int=0, int=0, int=0, bool=0>, int=16, bool=0, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_base_a<xmma_cudnn::Volta_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Volta<int=0>, int=64, int=128, int=32, int=2, int=2, int=1, int=1>, xmma_cudnn::implicit_gemm::Input_related<int=0, int=0, int=0, bool=0>, int=16, xmma_cudnn::Row, int=32, int=64>>, xmma_cudnn::implicit_gemm::fprop_indexed::Gmem_tile_c_t<xmma_cudnn::Volta_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Volta<int=0>, int=64, int=128, int=32, int=2, int=2, int=1, int=1>, int=16, xmma_cudnn::Fragment_c<xmma_cudnn::Volta_hmma_fp32_traits, xmma_cudnn::Cta_tile<xmma_cudnn::Volta<int=0>, int=64, int=128, int=32, int=2, int=2, int=1, int=1>, bool=0>>, xmma_cudnn::implicit_gemm::Input_related<int=0, int=0, int=0, bool=0>, int=1>>(xmma_cudnn::Volta_hmma_fp32_traitsParams)
                   26.13%  37.801ms      2000  18.900us  6.0160us  40.161us  void cudnn::ops::nchwToNhwcKernel<__half, __half, float, bool=0, bool=1, cudnnKernelDataType_t=0>(cudnn::ops::nchw2nhwc_params_t<float>, __half const *, __half*)
                    4.30%  6.2263ms      1000  6.2260us  5.9200us  11.200us  void cudnn::ops::nhwcToNchwKernel<__half, __half, float, bool=1, bool=0, cudnnKernelDataType_t=0>(cudnn::ops::nhwc2nchw_params_t<float>, __half const *, __half*)
                    1.24%  1.7880ms      1000  1.7880us  1.7270us  7.6480us  [CUDA memset]

total gpu_time: 144.6905 ms

W0706 08:37:25.901571 20400 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.4, Runtime API Version: 11.4
W0706 08:37:25.901605 20400 gpu_context.cc:306] device: 0, cuDNN Version: 8.2.

[paddle][conv2d] conv2d {
  run_tf: True
  run_torch: True
  atol: 0.01
  data_format: NCHW
  dilation: [1, 1]
  groups: 1
  padding: [1, 1]
  stride: 1
  x_shape: [16, 512, 7, 7]
  x_dtype: float16
  weight_shape: [512, 512, 3, 3]
  weight_dtype: float16
}
```

## PR 内容描述要求

单元测试内容需要和开发代码放在同一个 PR 提交，后续修改也需要基于此 PR。PR 内容描述测试部分需要明确描述下列内容：

1. 针对低精度数据类型的支持方法描述：概要说明计算精度是否对不同数据类型敏感，如 Transpose 算子的计算精度与数据类型无关

2. 扩展数据类型后算子的性能状况：给出不同数据类型下算子性能，如 FP32 和 FP16 的性能对比

3. PR 性能优化方案概述：如果扩展数据类型时，还对算子进行了性能优化，则需要描述优化方案

## 交流与改进

PR 的单测部分必须通过 Paddle 测试人员 review，确保完整覆盖了待测功能点后，会给予 approved。如果 review 过程中发现测试缺失和遗漏的测试点，会通过 GitHub 代码行 Comment 的和 Request Changes 的方式交流改进，待 PR 修改完毕后给予 approved。

## 后续维护

代码成功合入后，如果发现对框架造成了精度和性能下降影响，或者和部分功能存在严重冲突导致 Bug，会对代码进行 Revert 并通过 ISSUE 告知相关的开发者，请提交 PR 修复问题，并重新合入。
