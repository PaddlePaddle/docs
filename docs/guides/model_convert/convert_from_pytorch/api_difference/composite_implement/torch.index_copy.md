## [ 组合替代实现 ]torch.index_copy

### [torch.index_copy](https://pytorch.org/docs/stable/generated/torch.index_copy.html#torch.index_copy)

```python
torch.index_copy(input, dim, index, source, *, out=None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法，dim=0
y = torch.index_copy(input, 0, index, source)

# Paddle 写法
y = paddle.scatter(input, index, source)

# PyTorch 写法，dim>0
y = torch.index_copy(input, dim, index, source)

# Paddle 写法
times, temp_shape, temp_index = paddle.prod(paddle.to_tensor(input.shape[:dim])), input.shape, index
input, new_t = input.reshape([-1] + temp_shape[dim+1:]), source.reshape([-1] + temp_shape[dim+1:])
for i in range(1, times):
    temp_index= paddle.concat([temp_index, index+len(index)*i])
y = paddle.scatter(input, temp_index, new_t).reshape(temp_shape)
```
