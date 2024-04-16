## [ 参数完全一致 ] torch.Tensor.masked_scatter_

### [torch.Tensor.masked_scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter_.html?highlight=masked_scatter#torch.Tensor.masked_scatter_)

```python
torch.Tensor.masked_scatter_(mask, value)
```

### [paddle.Tensor.masked_scatter_]()

```python
paddle.Tensor.masked_scatter_(mask, value, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| mask     | mask          | 布尔张量，表示要填充的位置    |
| value     | value          | 用于填充目标张量的值    |
