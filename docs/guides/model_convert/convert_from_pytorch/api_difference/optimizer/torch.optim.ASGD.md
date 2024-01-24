## [ torch 参数更多 ]torch.optim.ASGD

### [torch.optim.ASGD](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html)

```python
torch.optim.ASGD(params,
                lr=0.01, 
                lambd=0.0001, 
                alpha=0.75, 
                t0=1000000.0, 
                weight_decay=0, 
                foreach=None, 
                maximize=False, 
                differentiable=False)
```

### [paddle.optimizer.ASGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/ASGD_cn.html#cn-api-paddle-optimizer-asgd)

```python
paddle.optimizer.ASGD(learning_rate=0.001,
                    batch_num=1,
                    parameters=None,
                    weight_decay=None,
                    grad_clip=None,
                    multi_precision=False,
                    name=None)
```

注：Pytorch 的 ASGD 是有问题的。
Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle        | 备注                                                                                                                    |
| ------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| params        | parameters          | 表示指定优化器需要优化的参数，仅参数名不一致                                                                                |
| lr            | learning_rate       | 学习率，用于参数更新的计算。参数默认值不一致, Pytorch 默认为 `0.0001`， Paddle 默认为 `0.001`，Paddle 需保持与 Pytorch 一致    |
| lambd         | -                   | 衰变项，与 weight_decay 功能重叠，暂无转写方式                                                                             |
| alpha         | -                   | eta 更新的 power，暂无转写方式                                                                                            |
| t0            | -                   | 开始求平均值的点，暂无转写方式                                                                                             |
| weight_decay  | weight_decay        | 权重衰减。参数默认值不一致, Pytorch 默认为 `0`， Paddle 默认为 `None`，Paddle 需保持与 Pytorch 一致                          |
| foreach       | -                   | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除                                         |
| maximize      | -                   | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式                                                               |
| differentiable| -                   | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除                                |
| -             | batch_num           | 完成一个 epoch 所需迭代的次数。 PyTorch 无此参数。假设样本总数为 all_size，Paddle 需将 batch_num 设置为 all_size / batch_size |
| -             | grad_clip           | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可                                                                      |
| -             | multi_precision     | 在基于 GPU 设备的混合精度训练场景中，该参数主要用于保证梯度更新的数值稳定性。 PyTorch 无此参数，Paddle 保持默认即可              |
| -             | name                | 一般情况下无需设置。 PyTorch 无此参数，Paddle 保持默认即可                                                                  |
