## [ 组合替代实现 ]torch.optim.lr_scheduler.StepLR

### [torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)

```python
torch.optim.lr_scheduler.StepLR(optimizer,
                                step_size,
                                gamma=0.1,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.StepDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/StepDecay_cn.html)

```python
paddle.optimizer.lr.StepDecay(learning_rate,
                                step_size,
                                gamma=0.1,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimize 类，Paddle 是 float 类。 |
| step_size     | step_size       | 表示学习率衰减轮数间隔。参数默认值一致。         |
| gamma     | gamma       | 表示学习率衰减率。参数默认值一致。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数默认值一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数默认值一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=sgd, step_size=2)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2)
sgd.set_lr_scheduler(scheduler)
```
