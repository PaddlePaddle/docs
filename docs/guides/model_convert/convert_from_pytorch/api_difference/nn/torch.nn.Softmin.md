
## [ 组合替代实现 ]torch.nn.Softmin

### [torch.nn.Softmin](https://pytorch.org/docs/stable/generated/torch.nn.Softmin.html#softmin)

```python
torch.nn.Softmin(dim=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
module = torch.nn.Softmin(dim=1)
module(-input)

# Paddle 写法
module = paddle.nn.Softmax(axis=1)
module(-input)
```
