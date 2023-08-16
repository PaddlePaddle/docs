## [ 组合替代实现 ]torch.optim.lr_scheduler.CyclicLR

### [torch.optim.lr_scheduler.CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html)

```python
torch.optim.lr_scheduler.CyclicLR(optimizer,
                                base_lr,
                                max_lr,
                                step_size_up=2000,
                                step_size_down=None,
                                mode='triangular',
                                gamma=1.0,
                                scale_fn=None,
                                scale_mode='cycle',
                                cycle_momentum=True,
                                base_momentum=0.8,
                                max_momentum=0.9,
                                last_epoch=- 1,
                                verbose=False)
```

### [paddle.optimizer.lr.CyclicLR](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/lr/CyclicLR_cn.html)

```python
paddle.optimizer.lr.CyclicLR(base_learning_rate,
                            max_learning_rate,
                            step_size_up,
                            step_size_down=None,
                            mode='triangular',
                            exp_gamma=1.,
                            scale_fn=None,
                            scale_mode='cycle',
                            last_epoch=- 1,
                            verbose=False)
```

两者 API 功能一致, 参数用法不一致，Pytorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | -       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 无对应参数。 |
| base_lr     | base_learning_rate       | 表示初始学习率，也是学习率变化的下边界。仅参数名不一致。           |
| max_lr     | max_learning_rate       | 表示最大学习率。仅参数名不一致。           |
| step_size_up     | step_size_up       | 表示学习率从初始学习率增长到最大学习率所需步数。PyTorch 默认值为 2000，Paddle 无默认值，Paddle 需保持与 Pytorch 一致。             |
| step_size_down     | step_size_down       | 表示学习率从最大学习率下降到初始学习率所需步数。若未指定，则其值默认等于 step_size_up。参数完全一致。             |
| mode     | mode       |  可以是 triangular、triangular2 或者 exp_range。参数完全一致。             |
| gamma     | exp_gamma       | 表示缩放函数中的常量。仅参数名不一致。             |
| scale_fn     | scale_fn       | 一个有且仅有单个参数的函数，且对于任意的输入 x，都必须满足 0 ≤ scale_fn(x) ≤ 1；如果该参数被指定，则会忽略 mode 参数。参数完全一致。             |
| scale_mode     | scale_mode       | cycle 或者 iterations，表示缩放函数使用 cycle 数或 iterations 数作为输入。参数完全一致。             |
| cycle_momentum     | -       | 如果"True"，动量反向循环 'base_momentum' 和 'max_momentum' 之间的学习率。Paddle 无此参数，暂无转写方式。             |
| base_momentum     | -       | 每个参数组的循环中的动量下边界。Paddle 无此参数，暂无转写方式。             |
| max_momentum     | -       | 每个参数组的循环中的动量上边界。Paddle 无此参数，暂无转写方式。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=sgd, base_lr=0.01, max_lr=0.1)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.CyclicLR(base_learning_rate=0.01, max_learning_rate=0.1, step_size_up=2000)
sgd.set_lr_scheduler(scheduler)
```
