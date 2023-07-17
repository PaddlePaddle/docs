## [ 组合替代实现 ]torch.optim.lr_scheduler.MultiStepLR

### [torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html)

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones,
                                gamma=0.1,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.MultiStepDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/MultiStepDecay_cn.html)

```python
paddle.optimizer.lr.MultiStepDecay(learning_rate,
                                milestones,
                                gamma=0.1,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，PyTorch 的 optimizer 参数是 torch.optim.Optimizer 类，Paddle 使用 paddle.optimizer.Optimizer.set_lr_scheduler 方法将 paddle.optimizer.Optimizer 和 paddle.optimizer.lr.LRScheduler 绑定，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 是 float 类。 |
| milestones     | milestones       | 表示轮数下标列表，必须递增。参数完全一致。         |
| gamma     | gamma       | 表示学习率衰减率。参数完全一致。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=sgd, milestones=[2,4,6])

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2,4,6])
sgd.set_lr_scheduler(scheduler)
```
