## [组合替代实现]torch.\_foreach_tan_

### [torch.\_foreach_tan_](https://pytorch.org/docs/stable/generated/torch._foreach_tan_.html#torch-foreach-tan)

```python
torch._foreach_tan_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_tan_(tensors)

# Paddle 写法
[paddle.tan_(x) for x in tensors]
```
