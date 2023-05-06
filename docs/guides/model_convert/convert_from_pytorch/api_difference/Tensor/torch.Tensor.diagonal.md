## [ 仅参数名不一致 ]torch.Tensor.diagonal

### [torch.Tensor.diagonal](https://pytorch.org/docs/stable/generated/torch.Tensor.diagonal.html?highlight=diagonal#torch.Tensor.diagonal)

```python
torch.Tensor.diagonal(offset=0, dim1=0, dim2=1)
```

### [paddle.Tensor.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#diagonal-offset-0-axis1-0-axis2-1-name-none)

```python
paddle.Tensor.diagonal(offset=0, axis1=0, axis2=1, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                       |
| ------- | ------------ | -------------------------------------------------------------------------- |
| offset  | offset       | 从指定的二维平面中获取对角线的位置，默认值为 0，既主对角线，仅参数名不同。 |
| dim1    | axis1        | 获取对角线的二维平面的第一维，默认值为 0，仅参数名不同。                   |
| dim2    | axis2        | 获取对角线的二维平面的第二维，默认值为 1，仅参数名不同。                   |
