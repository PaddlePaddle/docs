## [ torch参数更多 ]torch.distributions.constraints.Constraint

### [torch.distributions.constraints.Constraint](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints)

```python
torch.distributions.constraints.Constraint(is_discrete, event_dim)
```

### [paddle.distribution.constraint.Constraint](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/constraint.py)

```python
paddle.distribution.constraint.Constraint()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                        |
| ----------- | ------------ | --------------------------------------------------------------------------- |
| is_discrete | -            | 约束区域是否离散，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| event_dim   | -            | 最右侧维度的数量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
