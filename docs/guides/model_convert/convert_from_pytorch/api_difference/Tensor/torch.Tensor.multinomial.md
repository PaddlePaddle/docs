## [torch 参数更多]torch.Tensor.multinomial

### [torch.Tensor.multinomial](https://pytorch.org/docs/stable/generated/torch.Tensor.multinomial.html#torch.Tensor.multinomial)

```python
torch.Tensor.multinomial(num_samples, replacement=False, *, generator=None)
```

### [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multinomial_cn.html)

```python
paddle.multinomial(x, num_samples=1, replacement=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                |
| ----------- | ------------ | ----------------------------------------------------------------------------------- |
| num_samples | num_samples  | 采样的次数。                                                                        |
| replacement | replacement  | 是否是可放回的采样。                                                                |
| generator   | -            | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
