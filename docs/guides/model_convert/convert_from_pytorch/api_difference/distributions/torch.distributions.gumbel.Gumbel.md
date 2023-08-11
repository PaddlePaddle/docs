## [torch 参数更多]torch.distributions.gumbel.Gumbel

### [torch.distributions.gumbel.Gumbel](https://pytorch.org/docs/stable/distributions.html#torch.distributions.gumbel.Gumbel)

```python
torch.distributions.gumbel.Gumbel(loc, scale, validate_args=None)
```

### [paddle.distribution.Gumbel](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Gumbel_cn.html#gumbel)

```python
paddle.distribution.Gumbel(loc, scale)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| loc           | loc          | 耿贝尔分布位置参数。                                                    |
| scale         | scale        | 耿贝尔分布尺度参数。                                                    |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
