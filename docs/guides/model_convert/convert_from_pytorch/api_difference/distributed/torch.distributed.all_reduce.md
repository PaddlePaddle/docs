## [参数不一致]torch.distributed.all_reduce

### [torch.distributed.all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)

```python
torch.distributed.all_reduce(tensor, op=<torch.distributed.distributed_c10d.ReduceOp object>, group=None, async_op=False)
```

### [paddle.distributed.all_reduce](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/all_reduce_cn.html)

```python
paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                            |
| -------- | ------------ | --------------------------------------------------------------- |
| tensor   | tensor       | 操作的输入 Tensor。                                             |
| op       | op           | 归约的具体操作。                                                |
| group    | group        | 工作的进程组编号。                                              |
| async_op | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
