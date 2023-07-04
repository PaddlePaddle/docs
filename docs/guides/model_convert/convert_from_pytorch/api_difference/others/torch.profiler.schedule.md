## [仅参数名不一致]torch.profiler.schedule

### [torch.profiler.schedule](https://pytorch.org/docs/1.13/profiler.html#torch.profiler.schedule)

```python
torch.profiler.schedule(*, wait, warmup, active, repeat=0, skip_first=0)
```

### [paddle.profiler.make_scheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/profiler/make_scheduler_cn.html)

```python
paddle.profiler.make_scheduler(*, closed, ready, record, repeat=0, skip_first=0)
```

两者功能一致，参数名不一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                |
| ---------- | ------------ | ------------------------------------------------------------------- |
| wait       | closed       | 处于 wait/ProfilerState.CLOSED 状态的 step 数量，仅参数名不一致。   |
| warmup     | ready        | 处于 warmup/ProfilerState.READY 状态的 step 数量，仅参数名不一致。  |
| active     | record       | 处于 active/ProfilerState.RECORD 状态的 step 数量，仅参数名不一致。 |
| repeat     | repeat       | 重复次数。                                                          |
| skip_first | skip_first   | 首次 skip 步骤数量。                                                |
