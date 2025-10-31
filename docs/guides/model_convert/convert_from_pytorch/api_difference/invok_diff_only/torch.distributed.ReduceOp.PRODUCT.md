## [ 仅 API 调用方式不一致 ]torch.distributed.ReduceOp.PRODUCT

### [torch.distributed.ReduceOp.PRODUCT](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp.PRODUCT)

```python
torch.distributed.ReduceOp.PRODUCT
```

### [paddle.distributed.ReduceOp.PROD](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp/PROD_cn.html#paddle/distributed/ReduceOp/PROD_cn#cn-api-paddle-distributed-ReduceOp-PROD)

```python
paddle.distributed.ReduceOp.PROD
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
torch.distributed.reduce_scatter(
    data1,
    [data1, data2],
    op=torch.distributed.ReduceOp.PRODUCT,
    group=None,
    async_op=False
)

# Paddle 写法
paddle.distributed.reduce_scatter(
    tensor=data1,
    tensor_list=[data1, data2],
    op=paddle.distributed.ReduceOp.PROD,
    group=None,
    sync_op=not False,
)
```
