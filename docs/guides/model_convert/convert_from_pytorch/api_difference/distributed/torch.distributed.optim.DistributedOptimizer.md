## [ 输入参数用法不一致 ]torch.distributed.optim.DistributedOptimizer

### [torch.distributed.optim.DistributedOptimizer](https://pytorch.org/docs/stable/distributed.optim.html)

```python
torch.distributed.optim.DistributedOptimizer(optimizer_class, params_rref, *args, **kwargs)
```

### [paddle.distributed.fleet.distributed_optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/fleet/Fleet_cn.html#cn-api-paddle-distributed-fleet-fleet)

```python
paddle.distributed.fleet.distributed_optimizer(optimizer, strategy=None)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射


| PyTorch         | PaddlePaddle | 备注                                                        |
| --------------- | ------------ | ----------------------------------------------------------- |
| optimizer_class | optimizer    | 优化器。                                                    |
| params_rref     | -            | 初始化方法，paddle 无此参数，需要转写方式。                 |
| args            | -            | 优化器实例化参数。                                          |
| kwargs          | -            | 优化器实例化参数                                            |
| -               | strategy     | 用于切分优化器状态，PyTorch 无此参数，Paddle 保持默认即可。 |
