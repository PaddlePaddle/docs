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
(1,)
```

```python
# PyTorch 写法
torch.Size([2, 3])

# Paddle 写法
(2, 3)
```

```python
# PyTorch 写法
torch.Size([2, 3]).count(2)

# Paddle 写法
(2, 3).count(2)
```

```python
# PyTorch 写法
torch.Size([2, 3]).index(3,0)

# Paddle 写法
(2, 3).index(3,0)
```

```python
# PyTorch 写法
torch.Size([2, 3]).numel()

# Paddle 写法
result = (2, 3)
out = 1
for x in result:
    out *= x
```
