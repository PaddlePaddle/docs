## [组合替代实现]torch.\_foreach_floor_

### [torch.\_foreach_floor_](https://pytorch.org/docs/stable/generated/torch._foreach_floor_.html#torch-foreach-floor)

```python
torch._foreach_floor_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_floor_(tensors)

# Paddle 写法
def foreach_operator_(func, tensors):
    result = []
    for x in tensors:
        result.append(paddle.assign(func(x), x))
    return result
foreach_operator_(tensors, paddle.floor)
```
