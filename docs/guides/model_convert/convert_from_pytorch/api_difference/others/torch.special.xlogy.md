## [ 组合替代实现 ]torch.special.xlogy

### [torch.special.xlogy](https://pytorch.org/docs/stable/special.html#torch.special.xlogy)

```python
torch.special.xlogy(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.xlogy(a, b)

# Paddle 写法
y = a * paddle.log(b)
```
