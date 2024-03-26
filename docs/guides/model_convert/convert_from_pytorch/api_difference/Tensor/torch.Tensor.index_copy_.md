## [ 组合替代实现 ]torch.Tensor.index_copy_

### [torch.Tensor.index_copy_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html)

```python
torch.Tensor.index_copy_(dim, index, source)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法，dim=0
y = x.index_copy_(0, index, source)

# Paddle 写法
y = x.scatter_(index, source)

# PyTorch 写法，dim>0
y = x.index_copy_(dim, index, source)

# Paddle 写法
times, temp_shape, temp_index = paddle.prod(paddle.to_tensor(x.shape[:dim])), x.shape, index
x, new_t = x.reshape([-1] + temp_shape[dim+1:]), source.reshape([-1] + temp_shape[dim+1:])
for i in range(1, times):
    temp_index= paddle.concat([temp_index, index+len(index)*i])
y = x.scatter_(temp_index, new_t).reshape_(temp_shape)
```
