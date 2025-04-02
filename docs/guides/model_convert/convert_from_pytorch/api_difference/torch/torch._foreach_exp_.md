## [组合替代实现]torch.\_foreach_exp_

### [torch.\_foreach_exp_](https://pytorch.org/docs/stable/generated/torch._foreach_exp_.html#torch._foreach_exp_)

```python
torch._foreach_exp_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_exp_(tensors)

# Paddle 写法
[paddle.assign(paddle.exp(x), x) for x in tensors]
```
