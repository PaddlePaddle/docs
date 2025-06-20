## [组合替代实现]torch.\_foreach_cos_

### [torch.\_foreach_cos_](https://pytorch.org/docs/stable/generated/torch._foreach_cos_.html#torch-foreach-cos)

```python
torch._foreach_cos_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_cos_(tensors)

# Paddle 写法
[paddle.cos_(x) for x in tensors]
```
