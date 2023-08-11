## [torch 参数更多]torch.distributions.transforms.Transform

### [torch.distributions.transforms.Transform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.Transform)

```python
torch.distributions.transforms.Transform(cache_size=0)
```

### [paddle.distribution.Transform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Transform_cn.html#transform)

```python
paddle.distribution.Transform()
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                       |
| ---------- | ------------ | -------------------------------------------------------------------------- |
| cache_size | -            | 表示 cache 大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
