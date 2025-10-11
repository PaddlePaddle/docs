## [ 组合替代实现 ]torch.optim.lr_scheduler.LRScheduler

### [torch.optim.lr_scheduler.LRScheduler](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler)

```python
torch.optim.lr_scheduler.LRScheduler(optimizer,
                                last_epoch=-1)
```

### [paddle.optimizer.lr.LRScheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/lr/LRScheduler_cn.html#lrscheduler)

```python
paddle.optimizer.lr.LRScheduler(learning_rate=0.1,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，PyTorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的 optimizer 类型是 torch.optim.Optimizer，Paddle 的 learning_rate 类型是 float，两者功能上不直接一致，但可通过设置 leaning_rate = optimizer.get_lr() 来对应一致。  |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| -     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。|

### 转写示例

```python
# PyTorch 写法
linear = torch.nn.Linear(10, 10)
class CustomScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self, optimizer: Optimizer, last_epoch=-1, verbose="deprecated"
    ):
    ...
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = CustomScheduler(optimizer=sgd, lr_lambda=lambda x:0.95**x)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
class CustomScheduler(paddle.optimizer.lr.LRScheduler):
    def __init__(
        self,
        learning_rate: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
    ...
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = CustomScheduler(learning_rate=sgd.get_lr(), lr_lambda=lambda x:0.95**x)
sgd.set_lr_scheduler(scheduler)
```
