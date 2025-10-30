## [ 仅 API 调用方式不一致 ]torch.cuda.reset_max_memory_allocated

### [torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.reset_max_memory_allocated.html#torch.cuda.reset_max_memory_allocated)

```python
torch.cuda.reset_max_memory_allocated(device=None)
```

### [paddle.device.cuda.reset_max_memory_allocated](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_max_memory_allocated_cn.html#paddle/device/cuda/reset_max_memory_allocated_cn#cn-api-paddle-device-cuda-reset_max_memory_allocated)

```python
paddle.device.cuda.reset_max_memory_allocated(device=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.reset_max_memory_allocated()

# Paddle 写法
result = paddle.device.cuda.reset_max_memory_allocated()
```
