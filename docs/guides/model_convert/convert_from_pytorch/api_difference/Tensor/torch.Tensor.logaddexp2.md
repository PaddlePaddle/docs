## [ 组合替代实现 ]torch.Tensor.logaddexp2

### [torch.Tensor.logaddexp2](https://pytorch.org/docs/stable/generated/torch.Tensor.logaddexp2.html#torch.Tensor.logaddexp2)

```python
torch.Tensor.logaddexp2(other)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = a.logaddexp2(b)

# Paddle 写法
y = paddle.log2(2 ** a + 2 ** b)
```
