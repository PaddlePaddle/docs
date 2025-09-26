## [组合替代实现]torch.\_foreach_exp

### [torch.\_foreach_exp](https://pytorch.org/docs/stable/generated/torch._foreach_exp.html#torch-foreach-exp)

```python
torch._foreach_exp(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_exp(tensors)

# Paddle 写法
[paddle.exp(x) for x in tensors]
```
