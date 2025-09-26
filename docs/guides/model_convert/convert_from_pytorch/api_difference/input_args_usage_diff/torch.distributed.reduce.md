## [ 输入参数用法不一致 ]torch.distributed.reduce

### [torch.distributed.reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce)

```python
torch.distributed.reduce(tensor, dst, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```

### [paddle.distributed.reduce](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/reduce_cn.html)

```python
paddle.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=None, sync_op=True)
```

两者功能一致但参数不一致，其中 PyTorch 的 async_op 与 Paddle 的 sync_op 用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| tensor   | tensor       | 表示操作的输入 Tensor。                           |
| dst      | dst          | 表示目标进程的 rank。                  |
| op       | op           | 表示归约的具体操作。                              |
| group    | group        | 表示执行该操作的进程组实例。                            |
| async_op | sync_op      | 表示是否异步或同步操作，两者功能相反，需要转写。 |


### 转写示例
#### async_op：表示该操作是否为异步操作
```python
# PyTorch 写法
torch.distributed.reduce(tensor=tensor, dst=dst, op=<RedOpType.SUM: 0>, group=None, async_op=False)

# Paddle 写法
paddle.distributed.reduce(tensor=tensor, dst=dst, op=ReduceOp.SUM, group=None, sync_op=True)
```
