## [ 仅 API 调用方式不一致 ]torch.Tensor.bernoulli
### [torch.Tensor.bernoulli](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli.html#torch.Tensor.bernoulli)
```python
torch.Tensor.bernoulli(p=None, *, generator=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html#bernoulli)
```python
paddle.bernoulli(x, p=None, name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，另外 PyTorch 相比 Paddle 支持更多其他参数。具体如下：
