## [torch 参数更多]torch.distributions.uniform.Uniform

### [torch.distributions.uniform.Uniform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.uniform.Uniform)

```python
torch.distributions.uniform.Uniform(low, high, validate_args=None)
```

### [paddle.distribution.Uniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Uniform_cn.html)

```python
paddle.distribution.Uniform(low, high, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| low           | low          | 均匀分布的下边界。                            |
| high          | high         | 均匀分布的上边界。                            |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
