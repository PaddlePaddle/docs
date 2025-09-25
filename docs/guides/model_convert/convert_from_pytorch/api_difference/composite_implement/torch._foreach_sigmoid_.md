## [组合替代实现]torch.\_foreach_sigmoid_

### [torch.\_foreach_sigmoid_](https://pytorch.org/docs/stable/generated/torch._foreach_sigmoid_.html#torch-foreach-sigmoid)

```python
torch._foreach_sigmoid_(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_sigmoid_(tensors)

# Paddle 写法
[x.sigmoid_() for x in tensors]
```
