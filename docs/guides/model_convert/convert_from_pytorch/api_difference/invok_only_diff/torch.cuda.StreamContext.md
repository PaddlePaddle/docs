## [ 仅 API 调用方式不一致 ]torch.cuda.StreamContext

### [torch.cuda.StreamContext](https://docs.pytorch.org/docs/stable/generated/torch.cuda.StreamContext.html#torch.cuda.StreamContext)

```python
torch.cuda.StreamContext(stream)
```

### [paddle.device.stream\_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/stream_guard_cn.html#paddle.device.stream_guard)

```python
paddle.device.stream_guard(stream)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
with torch.cuda.StreamContext(stream=s1):
    result = a + b

# Paddle 写法
with paddle.device.stream_guard(stream=s1):
    result = a + b
```
