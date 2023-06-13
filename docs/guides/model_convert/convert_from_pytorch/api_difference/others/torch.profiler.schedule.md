## [torch 参数更多]torch.profiler.schedule

### [torch.profiler.schedule](https://pytorch.org/docs/1.13/profiler.html#torch.profiler.schedule)

```python
torch.profiler.schedule(*, wait, warmup, active, repeat=0, skip_first=0)
```

### [paddle.profiler.make_scheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/profiler/make_scheduler_cn.html)

```python
paddle.profiler.make_scheduler(*, closed, ready, record, repeat=0, skip_first=0)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                         |
| ---------- | ------------ | ---------------------------------------------------------------------------- |
| wait       | -            | 处于 wait 状态的 step 数量，Paddle 无此参数，暂无转写方式。                               |
| warmup     | -            | 处于 warmup 状态的 step 数量，Paddle 无此参数，暂无转写方式。                             |
| active     | -            | 处于 active 状态的 step 数量，Paddle 无此参数，暂无转写方式。                             |
| repeat     | repeat       | 重复次数。                                                                   |
| skip_first | skip_first   | 首次 skip 步骤数量。                                                         |
| -          | closed       | 处于 ProfilerState.CLOSED 状态的 step 数量，PyTorch 无此参数，暂无转写方式。 |
| -          | ready        | 处于 ProfilerState.READY 状态的 step 数量，PyTorch 无此参数，暂无转写方式。  |
| -          | record       | 处于 ProfilerState.RECORD 状态的 step 数量，PyTorch 无此参数，暂无转写方式。 |

### 转写示例

#### 参数用法不同

```python
# PyTorch 写法:
torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)

# Paddle 写法:
paddle.profiler.make_scheduler(closed=1, ready=1, record=2, repeat=1)
```
