## [ torch 参数更多 ] torch.Tensor.bernoulli

### [torch.Tensor.bernoulli](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli.html#torch.Tensor.bernoulli)

```python
torch.Tensor.bernoulli(p=0.5, *, generator=None)
```

### [paddle.Tensor.bernoulli]()

```python
paddle.Tensor.bernoulli(name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| p             | -            | 伯努利参数 p，Paddle 无此参数，暂无转写方式。                                        |
| generator     | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |

### 转写示例

```python
# torch 写法
x = torch.tensor([0.2, 0.6, 0.8])
y = x.bernoulli()

# paddle 写法
x = paddle.to_tensor([0.2, 0.6, 0.8])
y = x.bernoulli()
```
