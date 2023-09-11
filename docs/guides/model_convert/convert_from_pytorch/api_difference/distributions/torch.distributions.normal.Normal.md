## [torch 参数更多]torch.distributions.normal.Normal

### [torch.distributions.normal.Normal](https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal)

```python
torch.distributions.normal.Normal(loc, scale, validate_args=None)
```

### [paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Normal_cn.html)

```python
paddle.distribution.Normal(loc, scale, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| loc           | loc          | 正态分布平均值。                              |
| scale         | scale        | 正态分布标准差。                              |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
