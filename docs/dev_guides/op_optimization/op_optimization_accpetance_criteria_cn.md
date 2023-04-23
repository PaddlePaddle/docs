# 算子性能优化 验收规范

## CI 通过性

提交至 Paddle repo 的 Pull Request（简称 PR），涉及到的相关检测 CI 必须全部 Pass。用来验证对之前功能点的兼容和影响，保障新合入代码对历史代码不产生影响。

新增代码必须要有相应的单测保障测试覆盖率达到准入要求（测试覆盖率（行覆盖率)90%）。

## 性能测试

[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)作为一套测试飞桨内算子性能的专业工具, 如下图所示能够输出各类 case 下的 OP 性能真实状态, 建议用其进行算子性能测试。经过性能优化，OP Benchmark 中全部 case 不能出现性能下降，需要通过列表，对比性能优化前后的 OP 性能情况。

```
===========================================================================
-- paddle version             : 0.0.0
-- paddle commit              : 9b7126d05987b725ad3fb31f31298218c860b2f5
-- benchmark commit           : a6ba32197d7b3adb1dcc95b803f8f0d7fa18322c
-- benchmark last update time : Wed Jun 15 02:45:49 2022 +0000
===========================================================================
run command: nvprof --profile-from-start off /work/.virtualenvs_cuda10.2/paddle_py38/bin/python /work/benchmark/api/dynamic_tests_v2/adaptive_avg_pool2d.py --api_name adaptive_avg_pool2d --task speed --framework paddle --testing_mode dynamic --json_file /work/benchmark/api/tests_v2/configs/adaptive_avg_pool2d.json --config_id 0 --backward False --use_gpu True --repeat 1000 --allow_adaptive_repeat True --profiler nvprof
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  437.44ms      1000  437.44us  422.98us  472.29us  void phi::funcs::KernelPool2D<phi::funcs::AvgPool<float>, float>(int, phi::funcs::AvgPool<float> const *, int, int, int, int, int, int, int, int, int, int, int, phi::funcs::FastDivModForPooling, float, bool, bool, phi::funcs::KernelPool2D<phi::funcs::AvgPool<float>, float>*, bool)

total gpu_time: 437.4400 ms

W0615 14:55:43.819144 28877 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.4, Runtime API Version: 10.2 , cuDNN Version: 7.6.

[paddle][adaptive_avg_pool2d] adaptive_avg_pool2d {
  run_tf: True
  run_torch: True
  data_format: NCHW
  output_size: [32, 32]
  x_shape: [4, 2048, 64, 128]
  x_dtype: float32
  atol: 1e-06
}
{"framework": "paddle", "version": "0.0.0", "name": "adaptive_avg_pool2d", "device": "GPU", "backward": false, "speed": {"repeat": 1000, "begin": 10, "end": 990, "total": 0.6467142883612185, "wall_time": 0, "total_include_wall_time": 0.6467142883612185, "gpu_time": 0.43744}, "parameters": "x (Variable) - dtype: float32, shape: [4, 2048, 64, 128]\ndata_format (string): NCHW\noutput_size (list): [32, 32]\n"}
```

## PR 内容描述要求

单元测试内容需要和开发代码放在同一个 PR 提交，后续修改也需要基于此 PR。PR 内容描述测试部分需要明确描述下列内容：

    1. 合入前 Paddle 中算子的性能现状

    2. 业内最优方案的算子性能现状

    3. PR 性能优化方案概述

    4. 优化前后算子性能对比表格

## OP 测试内容及单元测试要求

性能测试至少覆盖[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)中全部的 case 场景。OP 性能优化后，需要在 Paddle 单元测试中对 GPU Kernel 进行有效性和边界值测试。

## 交流与改进

PR 的单测部分必须 Paddle 测试人员 review，确保完整覆盖了待测功能点后，会给予 approved。如果 review 过程中发现测试缺失和遗漏的测试点，会通过 GitHub 代码行 Comment 的和 Request Changes 的方式交流改进，待 PR 修改完毕后给予 approved。

## 后续维护

代码成功合入后，如果发现对框架造成了性能下降影响，或者和部分功能存在严重冲突导致 Bug，会对代码进行 Revert 并通过 ISSUE 告知相关的开发者，请提交 PR 修复问题，并重新合入。
