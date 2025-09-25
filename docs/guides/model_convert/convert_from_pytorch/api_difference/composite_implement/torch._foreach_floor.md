## [组合替代实现]torch.\_foreach_floor

### [torch.\_foreach_floor](https://pytorch.org/docs/stable/generated/torch._foreach_floor.html)

```python
torch._foreach_floor(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_floor(tensors)

# Paddle 写法
[paddle.floor(x) for x in tensors]
```
