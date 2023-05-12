## [ 组合替代实现 ]torch.column_stack

### [torch.column_stack](https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch.column_stack)

```python
torch.column_stack(tensors, *, out=None)
```

按水平方向拼接张量; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.column_stack((a, b))

# Paddle 写法
if a.ndim == 1:
    y = paddle.stack((a, b), axis=1)
else:
    y = paddle.concat((a, b), axis=1)
```
