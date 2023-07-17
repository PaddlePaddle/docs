## [ 组合替代实现 ]torch.optim.lr_scheduler.CosineAnnealingLR

### [torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                T_max,
                                eta_min=0,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.CosineAnnealingDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/CosineAnnealingDecay_cn.html)

```python
paddle.optimizer.lr.CosineAnnealingDecay(learning_rate,
                                T_max,
                                eta_min=0,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，PyTorch 的 optimizer 参数是 torch.optim.Optimizer 类，Paddle 使用 paddle.optimizer.Optimizer.set_lr_scheduler 方法将 paddle.optimizer.Optimizer 和 paddle.optimizer.lr.LRScheduler 绑定，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 是 float 类。 |
| T_max     | T_max       | 表示训练的上限轮数，是余弦衰减周期的一半。参数完全一致。             |
| eta_min     | eta_min       | 表示学习率的最小值。参数完全一致。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=sgd, T_max=10)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10)
sgd.set_lr_scheduler(scheduler)
```
