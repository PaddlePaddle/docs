## [ 组合替代实现 ]torch.row_stack

### [torch.row_stack](https://pytorch.org/docs/master/generated/torch.row_stack.html#torch.row_stack)

```python
torch.row_stack(tensors, *, out=None)
```

按垂直方向拼接张量; Paddle 无此 API，需要组合实现。

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.row_stack((a, b), out=y)

# Paddle 写法
if a.ndim == 1:
    y = paddle.stack((a, b))
else:
    y = paddle.concat((a, b))
```
