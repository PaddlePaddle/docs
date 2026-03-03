## [ 仅 API 调用方式不一致 ]torch.cuda.reset_max_memory_cached

### [torch.cuda.reset_max_memory_cached](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_max_memory_cached.html#torch.cuda.reset_max_memory_cached)

```python
torch.cuda.reset_max_memory_cached(device=None)
```

### [paddle.device.cuda.reset\_max\_memory\_reserved](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/device/cuda/reset_max_memory_reserved_cn.html#paddle.device.cuda.reset_max_memory_reserved)

```python
paddle.device.cuda.reset_max_memory_reserved(device=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.reset_max_memory_cached()

# Paddle 写法
result = paddle.device.cuda.reset_max_memory_reserved()
```
