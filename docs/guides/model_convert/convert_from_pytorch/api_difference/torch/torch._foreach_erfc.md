## [组合替代实现]torch.\_foreach_erfc

### [torch.\_foreach_erfc](https://pytorch.org/docs/stable/generated/torch._foreach_erfc.html#torch-foreach-erfc)

```python
torch._foreach_erfc(self)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch._foreach_erfc(tensors)

# Paddle 写法
tuple([(1-paddle.erf(x)) for x in tensors])
```
