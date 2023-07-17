## [ 组合替代实现 ]torch.optim.lr_scheduler.OneCycleLR

### [torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)

```python
torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                    max_lr,
                                    total_steps=None,
                                    epochs=None,
                                    steps_per_epoch=None,
                                    pct_start=0.3,
                                    anneal_strategy='cos',
                                    cycle_momentum=True,
                                    base_momentum=0.85,
                                    max_momentum=0.95,
                                    div_factor=25.0,
                                    final_div_factor=10000.0,
                                    three_phase=False,
                                    last_epoch=- 1,
                                    verbose=False)
```

### [paddle.optimizer.lr.OneCycleLR](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/OneCycleLR_cn.html)

```python
paddle.optimizer.lr.OneCycleLR(max_learning_rate,
                            total_steps,
                            divide_factor=25.,
                            end_learning_rate=0.0001,
                            phase_pct=0.3,
                            anneal_strategy='cos',
                            three_phase=False,
                            last_epoch=-1,
                            verbose=False)
```

两者 API 功能一致, 参数用法不一致，PyTorch 的 optimizer 参数是 torch.optim.Optimizer 类，Paddle 使用 paddle.optimizer.Optimizer.set_lr_scheduler 方法将 paddle.optimizer.Optimizer 和 paddle.optimizer.lr.LRScheduler 绑定，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | -       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 无对应参数。 |
| max_lr     | max_learning_rate       | 表示最大学习率。参数完全一致。           |
| total_steps     | total_steps       | 训练过程中总的迭代数。PyTorch 默认值为 None，Paddle 无默认值。           |
| steps_per_epoch     | -       | 每个 epoch 训练的步数。若 PyTorch 的 total_steps 参数为 None，则此参数用来计算 total_steps。PyTorch 默认值为 None。Paddle 无此参数。           |
| epochs     | -       | 训练的 epochs 数。若 PyTorch 的 total_steps 参数为 None，则此参数用来计算 total_steps。PyTorch 默认值为 None，Paddle 无默认值。           |
| pct_start     | phase_pct       | 表示学习率从初始学习率增长到最大学习率所需迭代数占总迭代数的比例。仅参数名不同。             |
| anneal_strategy     | anneal_strategy       | 调整学习率的策略。必须是 ( cos , linear )其中之一。参数完全一致。             |
| cycle_momentum     | -       | 如果“True”，动量反向循环 'base_momentum' 和 'max_momentum' 之间的学习率。Paddle 无对应参数，暂无转写方式。             |
| base_momentum     | -       | 每个参数组的循环中的动量下边界。Paddle 无对应参数，暂无转写方式。             |
| max_momentum     | -       | 每个参数组的循环中的动量上边界。Paddle 无对应参数，暂无转写方式。             |
| div_factor     | divide_factor       | 该参数用于推断初始学习率，公式为 initial_learning_rate = max_learning_rate/divide_factor。仅参数名不同。             |
| final_div_factor     | -       | 通过 min_lr = initial_lr/final_div_factor 确定最小学习率。Paddle 无对应参数，暂无转写方式。             |
| -     | end_learning_rate       | 最小学习率，学习率变化的下边界。PyTorch 无对应参数，Paddle 可通过公式：min_lr = max_lr/(div_factor * final_div_factor) 计算得出并设置。             |
| three_phase     | three_phase       | 是否使用三阶段调度策略。参数完全一致。            |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, steps_per_epoch=20, epochs=10)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.OneCycleLR(max_learning_rate=0.01, total_steps=20*10, end_learning_rate=max_lr/(25*10000))
sgd.set_lr_scheduler(scheduler)
```
