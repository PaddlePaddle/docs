## [组合替代实现]torch.\_foreach_cosh_

### [torch.\_foreach_cosh_](https://pytorch.org/docs/stable/generated/torch._foreach_cosh_.html#torch-foreach-cosh)

```python
torch._foreach_cosh_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_cosh_(tensors)

# Paddle 写法
[paddle.cosh_(x) for x in tensors]
```
