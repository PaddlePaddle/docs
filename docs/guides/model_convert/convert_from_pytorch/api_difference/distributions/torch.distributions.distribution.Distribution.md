## [torch 参数更多]torch.distributions.distribution.Distribution

### [torch.distributions.distribution.Distribution](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

```python
torch.distributions.distribution.Distribution(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

### [paddle.distribution.Distribution](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Distribution_cn.html)

```python
paddle.distribution.Distribution(batch_shape， event_shape)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| batch_shape   | batch_shape  | 概率分布参数批量形状。                        |
| event_shape   | event_shape  | 多元概率分布维数形状。                        |
| validate_args | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
