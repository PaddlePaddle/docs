## [ 组合替代实现 ]torch.Tensor.nbytes

### [torch.Tensor.nbytes](https://pytorch.org/docs/stable/generated/torch.Tensor.nbytes.html)

```python
torch.Tensor.nbytes
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
y = a.nbytes

# Paddle 写法
y = paddle.numel(a) * a.element_size()
```
