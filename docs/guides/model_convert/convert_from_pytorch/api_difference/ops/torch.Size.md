## [组合替代实现]torch.Size

### [torch.Size](https://pytorch.org/docs/stable/jit_builtin_functions.html#supported-pytorch-functions)

```python
torch.Size(sizes)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.Size([1])

# Paddle 写法
tuple([1])
```

```python
# PyTorch 写法
torch.Size([2, 3])

# Paddle 写法
tuple([2, 3])
```
