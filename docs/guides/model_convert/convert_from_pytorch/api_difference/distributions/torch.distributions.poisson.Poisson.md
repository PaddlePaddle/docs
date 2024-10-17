## [torch 参数更多 ]torch.distributions.poisson.Poisson

### [torch.distributions.poisson.Poisson](https://pytorch.org/docs/stable/distributions.html#poisson)

```python
torch.distributions.poisson.Poisson(rate, validate_args=None)
```

### [paddle.distribution.Poisson](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Poisson_cn.html)

```python
paddle.distribution.Poisson(rate)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                                    |
| ------------- | ------------- | ----------------------------------------------------------------------- |
| rate | rate | 表示输入的参数。                                                        |
| validate_args | -             | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
