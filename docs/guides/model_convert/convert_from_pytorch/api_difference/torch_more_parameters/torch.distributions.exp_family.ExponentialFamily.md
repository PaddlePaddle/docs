## [torch 参数更多]torch.distributions.exp_family.ExponentialFamily

### [torch.distributions.exp_family.ExponentialFamily](https://pytorch.org/docs/stable/distributions.html#torch.distributions.exp_family.ExponentialFamily)

```python
torch.distributions.exp_family.ExponentialFamily(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

### [paddle.distribution.ExponentialFamily](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/ExponentialFamily_cn.html)

```python
paddle.distribution.ExponentialFamily(batch_shape, event_shape)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                          |
| ------------- | ------------- | --------------------------------------------- |
| batch_shape   | batch_shape | 概率分布参数批量形状。                        |
| event_shape   | event_shape   | 多元概率分布维数形状。                        |
| validate_args | -             | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
