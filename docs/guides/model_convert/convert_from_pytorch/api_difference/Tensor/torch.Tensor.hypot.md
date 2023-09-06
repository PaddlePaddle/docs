## [ 组合替代实现 ]torch.Tensor.hypot

### [torch.Tensor.hypot](https://pytorch.org/docs/stable/generated/torch.Tensor.hypot.html#torch.Tensor.hypot)

```python
torch.Tensor.hypot(other)
```

给定直角三角形的直角边，求斜边; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.hypot(b)

# Paddle 写法
y = (a**2 + b**2) ** (1/2)
```
