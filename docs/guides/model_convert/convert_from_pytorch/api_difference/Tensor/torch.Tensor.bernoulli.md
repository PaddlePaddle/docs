## [ torch 参数更多 ] torch.Tensor.bernoulli

### [torch.Tensor.bernoulli](https://pytorch.org/docs/1.13/generated/torch.Tensor.bernoulli.html#torch.Tensor.bernoulli)

```python
torch.Tensor.bernoulli(*, generator=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html#bernoulli)

```python
paddle.bernoulli(x, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| generator     | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |

### 转写示例

```python
# torch 写法
x = torch.tensor([0.2, 0.6, 0.8])
y = x.bernoulli()

# paddle 写法
x = paddle.to_tensor([0.2, 0.6, 0.8])
y = paddle.bernoulli(x)
```
