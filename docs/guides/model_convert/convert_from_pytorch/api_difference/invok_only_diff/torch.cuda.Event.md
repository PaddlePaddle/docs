## [ 仅 API 调用方式不一致 ]torch.cuda.Event

### [torch.cuda.Event](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event)

```python
torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

### [paddle.device.cuda.Event](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/Event_cn.html#paddle/device/cuda/Event_cn#cn-api-paddle-device-cuda-Event)

```python
paddle.device.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.Event(enable_timing=True)

# Paddle 写法
result = paddle.device.cuda.Event(enable_timing=True)
```
