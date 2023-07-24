## [ 组合替代实现 ]torch.Tensor.bernoulli_

### [torch.Tensor.bernoulli_](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli_.html#torch.Tensor.bernoulli_)

```python
torch.Tensor.bernoulli_(p=0.5, *, generator=None)
```
Paddle 无此 API，需要组合实现。

### 转写示例
#### p：输入概率，类型为 tensor 时
```python
# Pytorch 写法
input.bernoulli_(p=x)

# Paddle 写法
paddle.assign(paddle.bernoulli(paddle.broadcast_to(x, input.shape)), input)
```

#### p：输入概率，类型为 float 时
```python
# Pytorch 写法
input.bernoulli_(p=x)

# Paddle 写法
tensor = paddle.to_tensor([x])
paddle.assign(paddle.bernoulli(paddle.broadcast_to(tensor, input.shape)), input)
```
