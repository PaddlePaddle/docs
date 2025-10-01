## [ torch 参数更多 ]torch.optim.SGD

### [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

```python
torch.optim.SGD(params,
                lr,
                momentum=0,
                dampening=0,
                weight_decay=0,
                nesterov=False,
                maximize=False,
                foreach=None,
                differentiable=False)
```

### [paddle.optimizer.SGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/SGD_cn.html)

```python
paddle.optimizer.SGD(learning_rate=0.001,
                    parameters=None,
                    weight_decay=None,
                    grad_clip=None,
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。PyTorch 无默认值，Paddle 默认为`0.001`，Paddle 需保持与 PyTorch 一致。          |
| momentum     | -       | 动量因子。Paddle 无此参数，暂无转写方式。     |
| dampening    | -        | 抑制动量。Paddle 无此参数，暂无转写方式。  |
| weight_decay           | weight_decay     | 表示权重衰减系数。参数默认值不一致, PyTorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 PyTorch 一致。         |
| nesterov   | -   | 打开 nesterov 动量。Paddle 无此参数，暂无转写方式。                       |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| foreach     | -           | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除                                         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
