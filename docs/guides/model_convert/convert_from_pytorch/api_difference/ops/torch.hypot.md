## [ 组合替代实现 ]torch.hypot

### [torch.hypot](https://pytorch.org/docs/stable/generated/torch.hypot.html#torch.hypot)

```python
torch.hypot(input, other, *, out=None)
```

给定直角三角形的直角边，求斜边; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.hypot(a, b)

# Paddle 写法
y = (a**2 + b**2) ** (1/2)
```
