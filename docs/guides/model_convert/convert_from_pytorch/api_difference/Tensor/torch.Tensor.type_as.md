## [ 组合替代实现 ] torch.Tensor.type_as

### [torch.Tensor.type_as](https://pytorch.org/docs/stable/generated/torch.Tensor.type_as.html)

```python
torch.Tensor.type_as(tensor)
```

### [paddle.Tensor.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#astype-dtype)

```python
paddle.Tensor.astype(dtype=tensor.dtype)
```

Paddle 无此 API，需要组合实现。

### 转写示例
####
```python
# Pytorch 写法
x.type_as(a)

# Paddle 写法
x.astype(dtype=a.dtype)
```
