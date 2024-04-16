## [ 组合替代实现 ]torch.special.xlog1py

### [torch.special.xlog1py](https://pytorch.org/docs/stable/special.html#torch.special.xlog1py)

```python
torch.special.xlog1py(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.xlog1py(a, b)

# Paddle 写法
y = a * paddle.log1p(b)
```
