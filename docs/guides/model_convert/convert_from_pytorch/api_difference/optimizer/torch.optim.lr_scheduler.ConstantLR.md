## [ 组合替代实现 ]torch.optim.lr_scheduler.ConstantLR

### [torch.optim.lr_scheduler.ConstantLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html)

```python
torch.optim.lr_scheduler.ConstantLR(optimizer,
                                factor=0.3333333333333333,
                                total_iters=5,
                                last_epoch=-1,
                                verbose=False)
```

### [paddle.optimizer.lr.PiecewiseDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/PiecewiseDecay_cn.html)

```python
paddle.optimizer.lr.PiecewiseDecay(boundaries,
                                values,
                                last_epoch=-1,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，PyTorch 的 optimizer 参数是 torch.optim.Optimizer 类，Paddle 使用 paddle.optimizer.Optimizer.set_lr_scheduler 方法将 paddle.optimizer.Optimizer 和 paddle.optimizer.lr.LRScheduler 绑定，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | -       | torch.optim.Optimizer 类，Paddle 无此参数。 |
| factor     | values       | PyTorch 表示乘以学习率的因子，Paddle 表示学习率列表。需要进行转写，转写思路为：values=[factor*optimizer.lr, optimizer.lr]。         |
| total_iters     | boundaries      | PyTorch 表示衰减学习率的步数，Paddle 表示指定学习率的边界值列表。需要进行转写，转写思路为：boundaries = [total_iters]。             |
| last_epoch     | last_epoch       | 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。参数完全一致。       |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=sgd, factor=0.5, total_iters=3)

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.PiecewiseDecay(values=[0.5*0.5, 0.5], boundaries=[3])
sgd.set_lr_scheduler(scheduler)
```
