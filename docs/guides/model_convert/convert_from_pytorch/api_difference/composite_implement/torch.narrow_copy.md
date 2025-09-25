## [ 组合替代实现 ]torch.narrow_copy

### [torch.narrow_copy](https://pytorch.org/docs/stable/generated/torch.narrow_copy.html#torch.narrow_copy)
```python
torch.narrow_copy(input, dim, start, length, *, out=None)
```

Paddle 目前无此 API，需要组合替代实现

### 转写示例
``` python
# PyTorch 写法：
torch.narrow_copy(x, 0, 1, 2)

# Paddle 写法：
# Paddle 可通过设置 ends-starts=length 来实现 PyTorch 的 length 功能
paddle.assign(paddle.slice(x, [0], [1], [3]))
```
