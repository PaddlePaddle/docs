## [ 仅参数名不一致 ]torch.Tensor.bernoulli

### [torch.Tensor.bernoulli](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.bernoulli)

```python
torch.Tensor.bernoulli(*, generator=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html#paddle/bernoulli_cn#cn-api-paddle-bernoulli)

```python
paddle.bernoulli(x, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                    |
| ------------- | ------------ | ----------------------------------------------------------------------------- |
| self      | x  | 伯努利参数 Tensor，将调用 torch.Tensor 类方法的 self Tensor 传入。  |
| p         | p  | 可选，伯努利参数 p。 |
| generator | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
