## [torch 参数更多]torch.profiler.profile

### [torch.profiler.profile](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile)

```python
torch.profiler.profile(*, activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None, use_cuda=None)
```

### [paddle.profiler.Profiler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Profiler_cn.html)

```python
paddle.profiler.Profiler(*, targets=None, scheduler=None, on_trace_ready=None, record_shapes=False, profile_memory=False, timer_only=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch             | PaddlePaddle   | 备注                                                                                                                         |
| ------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| activities          | targets        | 指定性能分析所要分析的设备，PyTorch 为 torch.profiler.ProfilerActivity 类型，Paddle 为 paddle.profiler.ProfilerTarget 类型。 |
| schedule            | scheduler      | 如果是 Callable 对象，代表是性能分析器状态的调度器，仅参数名不一致。                                                         |
| on_trace_ready      | on_trace_ready | 处理性能分析器的回调函数。                                                                                                   |
| record_shapes       | record_shapes  | 如果设置为 True, 则会开启收集框架算子输入张量的 shape。                                                                      |
| profile_memory      | profile_memory | 如果设置为 True, 则会开启收集显存分析的数据。                                                                                |
| with_stack          | -              | 记录 source 信息，Paddle 无此参数，暂无转写方式。                                                                            |
| with_flops          | -              | 使用公式来估计浮点计算，Paddle 无此参数，暂无转写方式。                                                                      |
| with_modules        | -              | 记录模块层次，Paddle 无此参数，暂无转写方式。                                                                                |
| experimental_config | -              | 实验性特征配置，Paddle 无此参数，暂无转写方式。                                                                              |
| use_cuda            | -              | 已废弃，Paddle 无此参数，暂无转写方式。                                                                                      |
| -                   | timer_only     | 如果设置为 True，将只统计模型的数据读取和每一个迭代所消耗的时间，而不进行性能分析，PyTorch 无此参数，Paddle 保持默认即可。   |

### 转写示例

#### 参数类型不同

```python
# PyTorch 写法:
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as p:
    for iter in range(10):
        p.step()

# Paddle 写法:
with paddle.profiler.Profiler(
    targets=[
        paddle.profiler.ProfilerTarget.CPU,
        paddle.profiler.ProfilerTarget.GPU
    ],
    scheduler=(2, 5),
    on_trace_ready = paddle.profiler.export_chrome_tracing('./log')
) as p:
    for iter in range(10):
        p.step()
```
