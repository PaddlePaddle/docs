## [torch 参数更多]torch.distributions.normal.Normal

### [torch.distributions.normal.Normal](https://pytorch.org/docs/1.13/distributions.html#torch.distributions.normal.Normal)

```python
torch.distributions.normal.Normal(loc, scale, validate_args=None)
```

### [paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Normal_cn.html)

```python
paddle.distribution.Normal(loc, scale, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| loc           | loc          | 正态分布平均值。                              |
| scale         | scale        | 正态分布标准差。                              |
| validate_args | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
