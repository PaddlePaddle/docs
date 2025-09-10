## [组合替代实现]torch.\_foreach_tan

### [torch.\_foreach_tan](https://pytorch.org/docs/stable/generated/torch._foreach_tan.html#torch-foreach-tan)

```python
torch._foreach_tan(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_tan(tensors)

# Paddle 写法
[paddle.tan(x) for x in tensors]
```
