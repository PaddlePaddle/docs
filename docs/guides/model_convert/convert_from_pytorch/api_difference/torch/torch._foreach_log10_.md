## [组合替代实现]torch.\_foreach_log10_

### [torch.\_foreach_log10_](https://pytorch.org/docs/stable/generated/torch._foreach_log10_.html#torch-foreach-log10)

```python
torch._foreach_log10_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_log10_(tensors)

# Paddle 写法
[paddle.log10_(x) for x in tensors]
```
