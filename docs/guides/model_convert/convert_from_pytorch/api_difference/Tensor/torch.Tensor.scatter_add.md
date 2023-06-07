## [组合替代实现]torch.Tensor.scatter_add

### [torch.Tensor.scatter_add](https://pytorch.org/docs/1.13/generated/torch.Tensor.scatter_add.html#torch.Tensor.scatter_add)

```python
torch.Tensor.scatter_add(dim, index, src)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
x.scatter_add(dim, index, src)

# Paddle 写法
x2 = paddle.zeros(x.shape)
y = x + x2.put_along_axis(index, value, axis)
```
