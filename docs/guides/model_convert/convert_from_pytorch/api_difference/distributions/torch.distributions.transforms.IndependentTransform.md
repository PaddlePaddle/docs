## [ torch 参数更多 ]torch.distributions.transforms.IndependentTransform

### [torch.distributions.transforms.IndependentTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.IndependentTransform)

```python
torch.distributions.transforms.IndependentTransform(base_transform, reinterpreted_batch_ndims, cache_size=0)
```

### [paddle.distribution.IndependentTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/IndependentTransform_cn.html#independenttransform)

```python
paddle.distribution.IndependentTransform(base, reinterpreted_batch_rank)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                   | PaddlePaddle             | 备注 |
| ------------------------- | ------------------------ | -- |
| base_transform            | base                     | 基础变换。 |
| reinterpreted_batch_ndims | reinterpreted_batch_rank | 被扩展为事件维度的最右侧批维度数量，需大于 0。 |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
