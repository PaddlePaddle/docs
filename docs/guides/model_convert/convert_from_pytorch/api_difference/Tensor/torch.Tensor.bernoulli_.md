## [ torch 参数更多 ]torch.Tensor.bernoulli_

### [torch.Tensor.bernoulli_](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli_.html#torch.Tensor.bernoulli_)

```python
torch.Tensor.bernoulli_(p=0.5, *, generator=None)
```

### [paddle.Tensor.bernoulli_](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/tensor.py)

```python
paddle.Tensor.bernoulli_(p=0.5, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：


### 参数映射

| PyTorch       | PaddlePaddle | 备注                    |
| ------------- | ------------ | ----------------------------------------------------------------------------- |
| p         | p  | 可选，伯努利参数 p。 |
| generator | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
