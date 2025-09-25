## [组合替代实现]torch.\_foreach_lgamma

### [torch.\_foreach_lgamma](https://pytorch.org/docs/stable/generated/torch._foreach_lgamma.html#torch-foreach-lgamma)

```python
torch._foreach_lgamma(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_lgamma(tensors)

# Paddle 写法
[paddle.lgamma(x) for x in tensors]
```
