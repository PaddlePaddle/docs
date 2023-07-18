## [ torch 参数更多 ]torch.optim.Adagrad

### [torch.optim.Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)

```python
torch.optim.Adagrad(params,
             lr=0.01,
             lr_decay=0,
             weight_decay=0,
             initial_accumulator_value=0,
             eps=1e-10,
             maximize=False,
             differentiable=False)
```

### [paddle.optimizer.Adagrad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Adagrad_cn.html)

```python
paddle.optimizer.Adagrad(learning_rate,
                          epsilon=1e-06,
                          parameters=None,
                          weight_decay=None,
                          grad_clip=None,
                          name=None,
                          initial_accumulator_value=0.0)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不同。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。参数默认值不一致, Pytorch 默认为`0.01`， Paddle 为必选参数，Paddle 需保持与 Pytorch 一致。                          |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, Pytorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 Pytorch 一致。         |
| initial_accumulator_value   | initial_accumulator_value   | 表示 moment 累加器的初始值，参数完全一致。                       |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值，参数默认值不一致, Pytorch 默认为`1e-10`， Paddle 为`1e-6`，Paddle 需保持与 Pytorch 一致。                           |
| lr_decay           | -     | 学习率衰减系数。Paddle 无此参数，暂无转写方式。         |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
