## [ 仅 API 调用方式不一致 ]torch.Tensor.bernoulli

### [torch.Tensor.bernoulli](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.bernoulli.html#torch.Tensor.bernoulli)

```python
torch.Tensor.bernoulli(p=0.5, *, generator=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html#paddle.bernoulli)

```python
paddle.bernoulli(x, p=None, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.bernoulli()

# Paddle 写法
result = paddle.bernoulli(a)
```
