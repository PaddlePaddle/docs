## [ 组合替代实现 ]torch.row_stack

### [torch.row_stack](https://pytorch.org/docs/master/generated/torch.row_stack.html#torch.row_stack)

```python
torch.row_stack(tensors, *, out=None)
```

按垂直方向拼接张量; Paddle 无此 API，需要组合实现。

```python
import paddle

def vstack(tensors, out=None):
    if a.ndim == 1:
        out = paddle.stack((a,b))
    else:
        out = paddle.concat((a,b))
    return out
```
