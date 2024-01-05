## [torch 参数更多]torch.distributions.laplace.Laplace

### [torch.distributions.laplace.Laplace](https://pytorch.org/docs/stable/distributions.html#torch.distributions.laplace.Laplace)

```python
torch.distributions.laplace.Laplace(loc, scale, validate_args=None)
```

### [paddle.distribution.Laplace](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Laplace_cn.html#laplace)

```python
paddle.distribution.Laplace(loc, scale)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| loc           | loc          | 拉普拉斯分布位置参数。                                                  |
| scale         | scale        | 拉普拉斯分布尺度参数。                                                  |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
