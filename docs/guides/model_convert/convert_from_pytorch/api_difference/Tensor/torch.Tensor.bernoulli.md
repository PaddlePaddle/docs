## [ torch 参数更多 ] torch.Tensor.bernoulli

### [torch.Tensor.bernoulli](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli.html#torch.Tensor.bernoulli)

```python
torch.Tensor.bernoulli(p=None, *, generator=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html#bernoulli)

```python
paddle.bernoulli(x, p=None, name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，另外 PyTorch 相比 Paddle 支持更多其他参数。具体如下：


### 参数映射

| PyTorch       | PaddlePaddle | 备注                    |
| ------------- | ------------ | ----------------------------------------------------------------------------- |
| self      | x  | 伯努利参数 Tensor，将调用 torch.Tensor 类方法的 self Tensor 传入。  |
| p         | p  | 可选，伯努利参数 p。 |
| generator | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |


### 转写示例
#### self：调用类方法的 Tensor
```python
# PyTorch 写法
x.bernoulli()

# Paddle 写法
paddle.bernoulli(x)
```
