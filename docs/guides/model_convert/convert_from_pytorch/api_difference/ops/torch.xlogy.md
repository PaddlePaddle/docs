## [ 组合替代实现 ]torch.xlogy

### [torch.xlogy](https://pytorch.org/docs/stable/generated/torch.xlogy.html#torch.xlogy)

```python
torch.xlogy(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.xlogy(a, b)

# Paddle 写法
y = a * paddle.log(b)
```
