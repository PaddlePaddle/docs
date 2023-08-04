## [ 组合替代实现 ]torch.optim.lr_scheduler.ReduceLROnPlateau

### [torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                        mode='min',
                                        factor=0.1,
                                        patience=10,
                                        threshold=0.0001,
                                        threshold_mode='rel',
                                        cooldown=0,
                                        min_lr=0,
                                        eps=1e-08,
                                        verbose=False)
```

### [paddle.optimizer.lr.ReduceOnPlateau](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/ReduceOnPlateau_cn.html)

```python
paddle.optimizer.lr.ReduceOnPlateau(learning_rate,
                                mode='min',
                                factor=0.1,
                                patience=10,
                                threshold=1e-4,
                                threshold_mode='rel',
                                cooldown=0,
                                min_lr=0,
                                epsilon=1e-8,
                                verbose=False)
```

两者 API 功能一致, 参数用法不一致，Pytorch 是 Scheduler 实例持有 Optimizer 实例，Paddle 是 Optimizer 实例持有 Scheduler 实例。由于持有关系相反，因此 Paddle 使用 Optimizer.set_lr_scheduler 来设置这种持有关系。具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| optimizer     | learning_rate       | PyTorch 的是 torch.optim.Optimizer 类，Paddle 是 float 类。 |
| mode     | mode       | 'min' 和 'max' 之一。通常情况下，为 'min'，此时当 loss 停止下降时学习率将衰减。参数完全一致。         |
| factor     | factor       | 表示学习率衰减的比例。参数完全一致。             |
| patience     | patience       |  当 loss 连续 patience 个 epoch 没有下降(对应 mode: 'min')或上升(对应 mode: 'max')时，学习率才会衰减。参数完全一致。       |
| threshold     | threshold       | threshold 和 threshold_mode 两个参数将会决定 loss 最小变化的阈值。小于该阈值的变化将会被忽视。参数完全一致。             |
| threshold_mode     | threshold_mode       | 'rel' 和 'abs' 之一。在 'rel' 模式下，loss 最小变化的阈值是 last_loss * threshold，其中 last_loss 是 loss 在上个 epoch 的值。在 'abs' 模式下，loss 最小变化的阈值是 threshold。参数完全一致。             |
| cooldown     | cooldown       | 在学习率每次衰减之后，会进入时长为 cooldown 个 step 的冷静期。参数完全一致。             |
| min_lr     | min_lr       | 最小的学习率。衰减后的学习率最低下界限。参数完全一致。             |
| eps     | epsilon       |  如果新旧学习率间的差异小于 epsilon，则不会更新。仅参数名不一致。             |
| verbose     | verbose       | 如果是 True，则在每一轮更新时在标准输出 stdout 输出一条信息。参数完全一致。  |

### 转写示例
```python
# Pytorch 写法
linear = torch.nn.Linear(10, 10)
sgd = torch.optimizer.SGD(lr=0.5, parameters=linear.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, 'min')

# Paddle 写法
linear = paddle.nn.linear(10, 10)
sgd = paddle.optimizer.SGD(learning_rate=0.5, parameters=linear.parameters())
scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=sgd.get_lr(), 'min')
sgd.set_lr_scheduler(scheduler)
```
