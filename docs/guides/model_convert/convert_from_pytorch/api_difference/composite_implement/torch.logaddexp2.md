## [ 组合替代实现 ]torch.logaddexp2

### [torch.logaddexp2](https://pytorch.org/docs/stable/generated/torch.logaddexp2.html#torch.logaddexp2)

```python
torch.logaddexp2(input, other, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.logaddexp2(a, b)

# Paddle 写法
y = paddle.log2(2 ** a + 2 ** b)
```
