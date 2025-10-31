## [ 仅 API 调用方式不一致 ]torch.distributed.ReduceOp

### [torch.distributed.ReduceOp](https://pytorch.org/docs/stable/generated/torch.distributed.ReduceOp.html#torch.distributed.ReduceOp)

```python
torch.distributed.ReduceOp
```

### [paddle.distributed.ReduceOp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html#paddle/distributed/ReduceOp_cn#cn-api-paddle-distributed-ReduceOp)

```python
paddle.distributed.ReduceOp
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
torch.distributed.init_process_group(backend="nccl")
print(torch.distributed.ReduceOp)

# Paddle 写法
paddle.distributed.init_parallel_env()
print(paddle.distributed.ReduceOp)
```
