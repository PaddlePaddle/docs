## [ 仅 API 调用方式不一致 ]torch.cuda.set_stream

### [torch.cuda.set_stream](https://pytorch.org/docs/stable/generated/torch.cuda.set_stream.html#torch.cuda.set_stream)

```python
torch.cuda.set_stream(stream)
```

### [paddle.device.set_stream](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/set_stream_cn.html#paddle/device/set_stream_cn#cn-api-paddle-device-set_stream)

```python
paddle.device.set_stream(stream)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.set_stream(stream)

# Paddle 写法
result = paddle.device.set_stream(stream)
```
