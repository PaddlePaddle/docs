## [torch 参数更多]torch.distributions.dirichlet.Dirichlet

### [torch.distributions.dirichlet.Dirichlet](https://pytorch.org/docs/stable/distributions.html#torch.distributions.dirichlet.Dirichlet)

```python
torch.distributions.dirichlet.Dirichlet(concentration, validate_args=None)
```

### [paddle.distribution.Dirichlet](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Dirichlet_cn.html)

```python
paddle.distribution.Dirichlet(concentration)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                          |
| ------------- | ------------- | --------------------------------------------- |
| concentration | concentration | 浓度参数，即公式中 α 参数。                   |
| validate_args | -             | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
