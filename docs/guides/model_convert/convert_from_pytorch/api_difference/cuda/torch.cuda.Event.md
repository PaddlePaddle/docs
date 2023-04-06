## [参数完全一致]torch.cuda.Event

### [torch.cuda.Event](https://pytorch.org/docs/1.13/generated/torch.cuda.Event.html#torch.cuda.Event)

```python
torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

### [paddle.device.cuda.Event](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/cuda/Event_cn.html#event)

```python
paddle.device.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

功能一致，参数完全一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| enable_timing        | enable_timing            | 表示是否需要统计时间。默认值为 False。                                     |
| blocking| blocking        | 表示 wait()函数是否被阻塞。默认值为 False。       |
| interprocess| interprocess        | 表示是否能在进程间共享。默认值为 False。       |
