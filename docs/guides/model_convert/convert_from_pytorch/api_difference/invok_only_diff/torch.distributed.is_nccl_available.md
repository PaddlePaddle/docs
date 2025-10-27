## [ 仅 API 调用方式不一致 ]torch.distributed.is_nccl_available

### [torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available)

```python
torch.distributed.is_nccl_available()
```

### [paddle.core.is_compiled_with_nccl](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/core/is_compiled_with_nccl_cn.html#paddle/core/is_compiled_with_nccl_cn#cn-api-paddle-core-is_compiled_with_nccl)

```python
paddle.core.is_compiled_with_nccl()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.distributed.is_nccl_available()

# Paddle 写法
result = paddle.core.is_compiled_with_nccl()
```
