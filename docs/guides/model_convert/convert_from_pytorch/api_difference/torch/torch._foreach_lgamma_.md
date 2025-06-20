## [组合替代实现]torch.\_foreach_lgamma_

### [torch.\_foreach_lgamma_](https://pytorch.org/docs/stable/generated/torch._foreach_lgamma_.html#torch-foreach-lgamma)

```python
torch._foreach_lgamma_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_lgamma_(tensors)

# Paddle 写法
[paddle.lgamma_(x) for x in tensors]
```
