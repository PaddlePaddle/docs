## [ 参数完全一致 ] torch.Tensor.masked_fill

### [torch.Tensor.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html?highlight=masked_fill#torch.Tensor.masked_fill)

```python
torch.Tensor.masked_fill(mask, value)
```

### [paddle.Tensor.masked_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#masked-fill-mask-value-name-non)

```python
paddle.Tensor.masked_fill(mask, value, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| mask     | mask          | 布尔张量，表示要填充的位置    |
| value     | value          | 用于填充目标张量的值    |
