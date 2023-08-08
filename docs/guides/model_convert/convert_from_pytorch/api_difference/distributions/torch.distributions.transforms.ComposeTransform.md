## [torch 参数更多]torch.distributions.transforms.ComposeTransform

### [torch.distributions.transforms.ComposeTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.ComposeTransform)

```python
torch.distributions.transforms.ComposeTransform(parts, cache_size=0)
```

### [paddle.distribution.ChainTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/ChainTransform_cn.html)

```python
paddle.distribution.ChainTransform(transforms)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                       |
| ---------- | ------------ | -------------------------------------------------------------------------- |
| parts      | transforms   | 输入的变换序列，仅参数名不一致。                                           |
| cache_size | -            | 表示 cache 大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
