## [ 仅参数名不一致 ] torch.Tensor.diagonal_scatter

### [torch.Tensor.diagonal_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.diagonal_scatter.html?highlight=diagonal_scatter#torch.Tensor.diagonal_scatter)

```python
torch.Tensor.diagonal_scatter(src, offset=0, dim1=0, dim2=1)
```

### [paddle.Tensor.diagonal_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#diagonal-scatter-x-y-offset-0-axis1-0-axis2-1-name-none)

```python
paddle.Tensor.diagonal_scatter(y, offset=0, axis1=0, axis2=1)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
| src     | y            | 用于嵌入的张量，仅参数名不一致。                     |
| offset  | offset       | 从指定的二维平面嵌入对角线的位置，默认值为 0，即主对角线。    |
| dim1    | axis1        | 对角线的第一个维度，默认值为 0，仅参数名不一致。    |
| dim2    | axis2        | 对角线的第二个维度，默认值为 1，仅参数名不一致。    |
