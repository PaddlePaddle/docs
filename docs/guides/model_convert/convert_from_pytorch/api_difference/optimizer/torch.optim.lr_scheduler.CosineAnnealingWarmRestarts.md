## [ 组合替代实现 ]torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

### [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
```

### [paddle.optimizer.lr.CosineAnnealingWarmRestarts](https://github.com/PaddlePaddle/Paddle/blob/d6ea911bd1bfda5604807eeb18318e71b395ac58/python/paddle/optimizer/lr.py#L2371)

```python
paddle.optimizer.lr.CosineAnnealingWarmRestarts(learning_rate,
                                        T_0,
                                        T_mult=1,
                                        eta_min=0,
                                        last_epoch=-1,
                                        verbose=False)
```

两者 API 功能一致, 参数用法不一致，Pytorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch    | PaddlePaddle  | 备注                                                                                                                                                                       |
| ---------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| optimizer  | learning_rate | PyTorch 的 optimizer 类型是 torch.optim.Optimizer，Paddle 的 learning_rate 类型是 float，两者功能上不直接一致，但可通过设置 leaning_rate = optimizer.get_lr() 来对应一致。 |
| T_0        | T_0           | 首次重启迭代数。                                                                                                                                                           |
| T_mult     | T_mult        | 重启后变量增加因子。                                                                                                                                                       |
| eta_min    | eta_min       | 表示学习率的最小值。                                                                                                                                                       |
| last_epoch | last_epoch    | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。                                                                                                                          |
| verbose    | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。                                                                                                              |

### 转写示例

```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=sgd, T_0=1)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.CosineAnnealingWarmRestarts(learning_rate=sgd.get_lr(), T_0=1)
sgd.set_lr_scheduler(scheduler)
```
