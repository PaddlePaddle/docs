## [ torch 参数更多 ]torch.optim.Adadelta

### [torch.optim.Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)

```python
torch.optim.Adadelta(params,
             rho=0.9,
             eps=1e-6,
             lr=1.0,
             weight_decay=0,
             maximize=False,
             differentiable=False)
```

### [paddle.optimizer.Adadelta](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Adadelta_cn.html#cn-api-paddle-optimizer-adadelta)

```python
paddle.optimizer.Adadelta(learning_rate=0.001,
                          epsilon=1e-06,
                          rho=0.95,
                          parameters=None,
                          weight_decay=None,
                          grad_clip=None,
                          name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                         |
| rho     | rho           | 表示衰减速率。参数默认值不一致, Pytorch 默认为`0.9`， Paddle 默认为`0.95`。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值，仅参数名不一致。                           |
| lr     | learning_rate       | 学习率，用于参数更新的计算。参数默认值不一致, Pytorch 默认为`1.0`， Paddle 默认为`0.001`。                          |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, Pytorch 默认为`0`， Paddle 默认为`None`。         |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
