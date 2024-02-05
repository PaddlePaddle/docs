## [ torch 参数更多 ]torch.distributions.transforms.AbsTransform

### [torch.distributions.transforms.AbsTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.AbsTransform)

```python
torch.distributions.transforms.AbsTransform(cache_size=0)
```

### [paddle.distribution.AbsTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/AbsTransform_cn.html#paddle.distribution.AbsTransform)

```python
paddle.distribution.AbsTransform()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
