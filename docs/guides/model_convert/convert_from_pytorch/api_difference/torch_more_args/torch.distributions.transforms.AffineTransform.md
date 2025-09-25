## [ torch 参数更多 ]torch.distributions.transforms.AffineTransform

### [torch.distributions.transforms.AffineTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.AffineTransform)

```python
torch.distributions.transforms.AffineTransform(loc, scale, event_dim=0, cache_size=0)
```

### [paddle.distribution.AffineTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/AffineTransform_cn.html#affinetransform)

```python
paddle.distribution.AffineTransform(loc, scale)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| loc        | loc          | 偏置参数。 |
| scale      | scale        | 缩放参数。 |
| event_dim  | -            | event_shape 可选尺寸。对于单随机变量为 0，对于向量分布为 1，对于矩阵分布为 2。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
