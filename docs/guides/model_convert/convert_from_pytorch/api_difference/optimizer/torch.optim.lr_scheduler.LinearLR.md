## [ 组合替代实现 ]torch.optim.lr_scheduler.LinearLR

### [torch.optim.lr_scheduler.LinearLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)

```python
torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False)
```

### [paddle.optimizer.lr.LinearLR](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/lr/LinearLR_cn.html#linearlr)

```python
paddle.optimizer.lr.LinearLR(learning_rate, total_steps, start_factor=1. / 3, end_factor=1.0, last_epoch=- 1, verbose=False)
```

两者 API 功能一致, 参数用法不一致，Pytorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch      | PaddlePaddle  | 备注                                                                                                                                                                       |
| ------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| optimizer    | learning_rate | PyTorch 的 optimizer 类型是 torch.optim.Optimizer，Paddle 的 learning_rate 类型是 float，两者功能上不直接一致，但可通过设置 leaning_rate = optimizer.get_lr() 来对应一致。 |
| start_factor | start_factor  | 初始学习率因子。                                                                                                                                                           |
| end_factor   | end_factor    | 最终学习率因子。                                                                                                                                                           |
| total_iters  | total_steps   | 学习率从初始学习率线性增长到最终学习率所需要的步数，仅参数名不一致。                                                                                                       |
| last_epoch   | last_epoch    | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。                                                                                                                          |
| verbose      | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。                                                                                                              |

### 转写示例

```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=sgd)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.LinearLR(learning_rate=sgd.get_lr(), total_steps=5)
sgd.set_lr_scheduler(scheduler)
```
