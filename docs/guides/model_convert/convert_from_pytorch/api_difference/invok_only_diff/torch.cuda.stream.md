## [ 仅 API 调用方式不一致 ]torch.cuda.stream

### [torch.cuda.stream](https://pytorch.org/docs/stable/generated/torch.cuda.stream.html#torch.cuda.stream)

```python
torch.cuda.stream(stream)
```

### [paddle.device.stream_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/stream_guard_cn.html#paddle/device/stream_guard_cn#cn-api-paddle-device-stream_guard)

```python
paddle.device.stream_guard(stream)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
with torch.cuda.stream(stream=s):
    result = data1 + data2

# Paddle 写法
with paddle.device.stream_guard(stream=s):
    result = data1 + data2

```
