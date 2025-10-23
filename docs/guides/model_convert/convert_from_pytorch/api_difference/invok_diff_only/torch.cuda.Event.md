## [ 仅 API 调用方式不一致 ]torch.cuda.Event
### [torch.cuda.Event](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event)
```python
torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

### [paddle.device.cuda.Event](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/Event_cn.html)
```python
paddle.device.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

功能一致，参数完全一致，具体如下：
