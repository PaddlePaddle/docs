## [组合替代实现]torch.\_foreach_acos_

### [torch.\_foreach_acos_](https://pytorch.org/docs/stable/generated/torch._foreach_acos_.html#torch-foreach-acos)

```python
torch._foreach_acos_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_acos_(tensors)

# Paddle 写法
[paddle.acos_(x) for x in tensors]
```
