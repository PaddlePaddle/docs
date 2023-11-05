## [ 参数完全一致 ] torch.Tensor.diagonal_scatter

### [torch.Tensor.diagonal_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.diagonal_scatter.html?highlight=diagonal_scatter#torch.Tensor.diagonal_scatter)

```python
torch.Tensor.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1)
```

### [paddle.Tensor.diagonal_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#diagonal-scatter-x-y-offset-0-axis1-0-axis2-1-name-none)

```python
paddle.Tensor.diagonal_scatter(x, y, offset=0, axis1=0, axis2=1)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| input     | x          | 输入张量，被嵌入的张量    |
| src     | y          | 用于嵌入的张量    |
| offset     | offset          | 偏移的对角线    |
| dim1     | axis1          | 对角线的第一个维度    |
| dim2     | axis2          | 对角线的第二个维度    |
