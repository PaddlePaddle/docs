## [组合替代实现]torch.\_foreach_trunc

### [torch.\_foreach_trunc](https://pytorch.org/docs/stable/generated/torch._foreach_trunc.html#torch-foreach-trunc)

```python
torch._foreach_trunc(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_trunc(tensors)

# Paddle 写法
[paddle.trunc(x) for x in tensors]
```
