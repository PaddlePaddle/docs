## [ 组合替代实现 ]torch.special.ndtr

### [torch.special.ndtr](https://pytorch.org/docs/stable/special.html#torch.special.ndtr)

```python
torch.special.ndtr(input, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = torch.special.ndtr(a)

# Paddle 写法
y = (paddle.erf(a/paddle.sqrt(paddle.to_tensor(2)))-paddle.erf(paddle.to_tensor(-float('inf'))))/2
```
