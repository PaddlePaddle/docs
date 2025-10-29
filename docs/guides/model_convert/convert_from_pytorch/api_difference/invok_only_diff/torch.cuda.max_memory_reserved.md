## [ 仅 API 调用方式不一致 ]torch.cuda.max_memory_reserved

### [torch.cuda.max_memory_reserved](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved)

```python
torch.cuda.max_memory_reserved(device=None)
```

### [paddle.device.cuda.max_memory_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/max_memory_reserved_cn.html#paddle/device/cuda/max_memory_reserved_cn#cn-api-paddle-device-cuda-max_memory_reserved)

```python
paddle.device.cuda.max_memory_reserved(device=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.max_memory_reserved()

# Paddle 写法
result = paddle.device.cuda.max_memory_reserved()
```
