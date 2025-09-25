## [组合替代实现]torch.\_foreach_neg_

### [torch.\_foreach_neg_](https://pytorch.org/docs/stable/generated/torch._foreach_neg_.html#torch-foreach-neg)

```python
torch._foreach_neg_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_neg_(tensors)

# Paddle 写法
[paddle.neg_(x) for x in tensors]
```
