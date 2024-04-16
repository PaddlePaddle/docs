## [torch 参数更多]torch.distributions.transforms.StickBreakingTransform

### [torch.distributions.transforms.StickBreakingTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StickBreakingTransform)

```python
torch.distributions.transforms.StickBreakingTransform(cache_size=0)
```

### [paddle.distribution.StickBreakingTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/StickBreakingTransform_cn.html)

```python
paddle.distribution.StickBreakingTransform()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                       |
| ---------- | ------------ | -------------------------------------------------------------------------- |
| cache_size | -            | 表示 cache 大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
