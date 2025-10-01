## [组合替代实现]torch.\_foreach_round_

### [torch.\_foreach_round_](https://pytorch.org/docs/stable/generated/torch._foreach_round_.html#torch-foreach-round)

```python
torch._foreach_round_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_round_(tensors)

# Paddle 写法
[x.round_() for x in tensors]
```
