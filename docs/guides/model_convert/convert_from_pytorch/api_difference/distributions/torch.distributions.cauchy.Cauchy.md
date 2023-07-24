## [torch 参数更多]torch.distributions.cauchy.Cauchy

### [torch.distributions.cauchy.Cauchy](https://pytorch.org/docs/stable/distributions.html#torch.distributions.cauchy.Cauchy)

```python
torch.distributions.cauchy.Cauchy(loc, scale, validate_args=None)
```

### [paddle.distribution.Cauchy](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Cauchy_cn.html)

```python
paddle.distribution.Cauchy(loc, scale, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| loc           | loc          | 定义分布峰值位置的位置参数。                                            |
| scale         | scale        | 最大值一半处的一半宽度的尺度参数。                                      |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
