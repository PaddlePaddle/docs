## [ 组合替代实现 ]torch.optim.lr_scheduler.MultiplicativeLR

### [torch.optim.lr_scheduler.MultiplicativeLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html)

```python
torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
                                lr_lambda,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.MultiplicativeDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/MultiplicativeDecay_cn.html)

```python
paddle.optimizer.lr.MultiplicativeDecay(learning_rate,
                                lr_lambda,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，Pytorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 是 float 类。 |
| lr_lambda     | lr_lambda       | 表示 lambda 函数，其通过 epoch 计算出一个因子，该因子会乘以初始学习率。PyTorch 可以是 lambda 函数的列表，Paddle 只能表示 lambda 函数。当 PyTorch 是 lambda 函数的列表时，Paddle 暂无转写方式。   |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=sgd, lr_lambda=lambda x:0.95**x)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=sgd.get_lr(), lr_lambda=lambda x:0.95**x)
sgd.set_lr_scheduler(scheduler)
```
