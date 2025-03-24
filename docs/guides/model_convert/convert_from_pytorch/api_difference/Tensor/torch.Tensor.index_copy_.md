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
shape = x.shape
new_index = []
for i in range(0, np.prod(shape[:dim])):
    new_index.append(index + i * len(index))
new_index = paddle.concat(new_index)
new_x = x.reshape_([-1] + shape[dim + 1:])
new_source = source.reshape([-1] + shape[dim + 1:])
y = new_x.scatter_(new_index, new_source).reshape_(shape)
```
