## [torch 参数更多]torch.distributions.uniform.Uniform

### [torch.distributions.uniform.Uniform](https://pytorch.org/docs/1.13/distributions.html#torch.distributions.uniform.Uniform)

```python
torch.distributions.uniform.Uniform(low, high, validate_args=None)
```

### [paddle.distribution.Uniform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Uniform_cn.html)

```python
paddle.distribution.Uniform(low, high, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| low           | low          | 均匀分布的下边界。                            |
| high          | high         | 均匀分布的上边界。                            |
| validate_args | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
