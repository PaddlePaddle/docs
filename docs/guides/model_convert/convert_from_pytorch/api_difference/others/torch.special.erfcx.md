## [ 组合替代实现 ]torch.special.erfcx

### [torch.special.erfcx](https://pytorch.org/docs/stable/special.html#torch.special.erfcx)

```python
torch.special.erfcx(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.special.erfcx(x)

# Paddle 写法
y = paddle.exp(x ** 2) * (1.0 - paddle.erf(x))
```
