## [ torch 参数更多 ]torch.distributions.transforms.SigmoidTransform

### [torch.distributions.transforms.SigmoidTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.SigmoidTransform)

```python
torch.distributions.transforms.SigmoidTransform(cache_size=0)
```

### [paddle.distribution.SigmoidTransform](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/SigmoidTransform_cn.html#sigmoidtransform)

```python
paddle.distribution.SigmoidTransform()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| cache_size | -            | 缓存大小，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
