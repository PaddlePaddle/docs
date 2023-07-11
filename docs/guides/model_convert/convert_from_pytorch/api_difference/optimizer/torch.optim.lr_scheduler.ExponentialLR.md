## [ 组合替代实现 ]torch.optim.lr_scheduler.ExponentialLR

### [torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html)

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                gamma,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.ExponentialDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/ExponentialDecay_cn.html)

```python
paddle.optimizer.lr.ExponentialDecay(learning_rate,
                                gamma,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimize 类，Paddle 是 float 类。 |
| gamma     | gamma       | 表示衰减率。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数默认值一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数默认值一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=sgd, gamma=0.5)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.5, gamma=0.5)
sgd.set_lr_scheduler(scheduler)
```
