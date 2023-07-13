## [ torch 参数更多 ]torch.optim.Adamax

### [torch.optim.Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html)

```python
torch.optim.Adamax(params,
                lr=0.002,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                maximize=False,
                differentiable=False)
```

### [paddle.optimizer.Adamax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Adamax_cn.html)

```python
paddle.optimizer.Adamax(learning_rate=0.001,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-08,
                        parameters=None,
                        weight_decay=None,
                        grad_clip=None,
                        name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不同。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。参数默认值不一致, Pytorch 默认为`0.002`，Pytorch 默认为`0.001`。                    |
| betas     | beta1、beta2       | 一阶矩估计的指数衰减率。Pytorch 为元祖形式，Paddle 为分开的两个参数。默认值分别一致。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值，参数默认值不一致, Pytorch 默认为`1e-10`， Paddle 为`1e-6`。    |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, Pytorch 默认为`0`， Paddle 默认为`None`。         |
| lr_decay           | -     | 学习率衰减系数。Paddle 无此参数，暂无转写方式。         |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，可直接删除。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
