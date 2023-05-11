## [ 组合替代实现 ]torch.Tensor.ldexp

### [torch.Tensor.ldexp](https://pytorch.org/docs/stable/generated/torch.Tensor.ldexp.html#torch.Tensor.ldexp)

```python
torch.Tensor.ldexp(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.ldexp(b)

# Paddle 写法
y = a * (2 ** b)
```
