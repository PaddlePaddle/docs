## [ 仅参数名不一致 ]torch.Tensor.diagonal

### [torch.diagonal](https://pytorch.org/docs/stable/generated/torch.diagonal.html?highlight=diagonal#torch.diagonal)

```python
torch.diagonal(input, offset=0, dim1=0, dim2=1)
```

### [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_cn.html#diagonal)

```python
paddle.diagonal(x, offset=0, axis1=0, axis2=1, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                   | PaddlePaddle              | 备注                                                                       |
| ------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| <center> input </center>  | <center> x </center>      | 输入 Tensor，仅参数名不同。                                                |
| <center> offset </center> | <center> offset </center> | 从指定的二维平面中获取对角线的位置，默认值为 0，既主对角线，仅参数名不同。 |
| <center> dim1 </center>   | <center> axis1 </center>  | 获取对角线的二维平面的第一维，默认值为 0，仅参数名不同。                   |
| <center> dim2 </center>   | <center> axis2 </center>  | 获取对角线的二维平面的第二维，默认值为 1，仅参数名不同。                   |
