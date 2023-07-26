## [torch 参数更多]torch.distributions.transformed_distribution.TransformedDistribution

### [torch.distributions.transformed_distribution.TransformedDistribution](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution)

```python
torch.distributions.transformed_distribution.TransformedDistribution(base_distribution, transforms, validate_args=None)
```

### [paddle.distribution.TransformedDistribution](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/TransformedDistribution_cn.html#transformeddistribution)

```python
paddle.distribution.TransformedDistribution(base, transforms)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle | 备注                                                                    |
| ----------------- | ------------ | ----------------------------------------------------------------------- |
| base_distribution | base         | 基础分布，仅参数名不一致。                                              |
| transforms        | transforms   | 变换序列。                                                              |
| validate_args     | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
