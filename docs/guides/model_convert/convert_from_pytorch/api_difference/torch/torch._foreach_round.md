## [组合替代实现]torch.\_foreach_round

### [torch.\_foreach_round](https://pytorch.org/docs/stable/generated/torch._foreach_round.html#torch-foreach-round)

```python
torch._foreach_round(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_round(tensors)

# Paddle 写法
[paddle.round(x) for x in tensors]
```
