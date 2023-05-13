## [ 组合替代实现 ]torch.ldexp

### [torch.ldexp](https://pytorch.org/docs/stable/generated/torch.ldexp.html#torch.ldexp)

```python
torch.ldexp(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.ldexp(a, b)

# Paddle 写法
y = a * (2 ** b)
```
