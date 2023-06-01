## [ 组合替代实现 ]torch.adjoint

### [torch.adjoint](https://pytorch.org/docs/stable/generated/torch.adjoint.html#torch.adjoint)
```python
torch.adjoint(input)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.adjoint(input)

# Paddle 写法
perm = list(range(input.ndim))
perm[-1], perm[-2] = perm[-2], perm[-1]
y = paddle.conj(paddle.transpose(input, perm=perm))
```
