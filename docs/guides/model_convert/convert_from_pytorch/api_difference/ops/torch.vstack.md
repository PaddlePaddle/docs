## [ 组合替代实现 ]torch.vstack

### [torch.vstack](https://pytorch.org/docs/stable/generated/torch.vstack.html#torch.vstack)

```python
torch.vstack(tensors, *, out=None)
```

按垂直方向拼接张量; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = torch.vstack((a, b))

# Paddle 写法
if a.ndim == 1:
    y = paddle.stack((a, b))
else:
    y = paddle.concat((a, b))
```
