## [torch 参数更多 ]torch.distributions.gamma.Gamma

### [torch.distributions.gamma.Gamma](https://pytorch.org/docs/stable/distributions.html#gamma)

```python
torch.distributions.gamma.Gamma(concentration, rate, validate_args=None)
```

### [paddle.distribution.Gamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Gamma_cn.html)

```python
paddle.distribution.Gamma(concentration, rate)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| concentration            | concentration           | 表示输入的参数。                                                        |
| rate            | rate           | 表示输入的参数。                                                        |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
