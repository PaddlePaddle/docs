## [ 输入参数用法不一致 ]torch.distributed.optim.DistributedOptimizer

### [torch.distributed.optim.DistributedOptimizer](https://pytorch.org/docs/stable/distributed.optim.html)

```python
torch.distributed.optim.DistributedOptimizer(optimizer_class, params_rref, *args, **kwargs)
```

### [paddle.distributed.shard_optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/shard_optimizer_cn.html)

```python
paddle.distributed.shard_optimizer(optimizer, shard_fn=None)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射


| PyTorch         | PaddlePaddle | 备注                                        |
| --------------- | ------------ | ------------------------------------------- |
| optimizer_class | optimizer    | 优化器。                                    |
| params_rref     | -            | 初始化方法，paddle 无此参数，暂无转写方式。 |
| args            | -            | 优化器实例化参数。                          |
| kwargs          | -            | 优化器实例化参数                            |
| -               | shard_fn     | 用于切分优化器状态。                        |
