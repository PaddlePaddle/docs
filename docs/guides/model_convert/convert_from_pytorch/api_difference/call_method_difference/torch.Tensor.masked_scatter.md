## [ 参数完全一致 ] torch.Tensor.masked_scatter

### [torch.Tensor.masked_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter.html?highlight=masked_scatter#torch.Tensor.masked_scatter)

```python
torch.Tensor.masked_scatter(mask, value)
```

### [paddle.Tensor.masked_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#masked-scatter-mask-value-name-non)

```python
paddle.Tensor.masked_scatter(mask, value, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| mask     | mask          | 布尔张量，表示要填充的位置    |
| value     | value          | 用于填充目标张量的值    |
