## [ 组合替代实现 ]torch.hstack

### [torch.hstack](https://pytorch.org/docs/stable/generated/torch.hstack.html#torch.hstack)

```python
torch.hstack(tensors, *, out=None)
```

按水平方向拼接张量; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.hstack((a, b))

# Paddle 写法
if a.ndim == 1:
    y = paddle.concat((a, b), axis=0)
else:
    y = paddle.concat((a, b), axis=1)
```
