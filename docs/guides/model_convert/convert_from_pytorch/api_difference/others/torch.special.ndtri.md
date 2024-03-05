## [ 组合替代实现 ]torch.special.ndtri

### [torch.special.ndtri](https://pytorch.org/docs/stable/special.html#torch.special.ndtri)

```python
torch.special.ndtri(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.ndtri(a)

# Paddle 写法
y = 2 ** (1/2) * paddle.erfinv(2*a-1)
```
