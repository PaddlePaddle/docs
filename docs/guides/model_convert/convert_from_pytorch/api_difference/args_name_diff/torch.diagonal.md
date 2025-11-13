## [ 仅参数名不一致 ]torch.diagonal
### [torch.diagonal](https://pytorch.org/docs/stable/generated/torch.diagonal.html?highlight=diagonal#torch.diagonal)
```python
torch.diagonal(input,
               offset=0,
               dim1=0,
               dim2=1)
```

### [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_cn.html#diagonal)
```python
paddle.diagonal(x,
                offset=0,
                axis1=0,
                axis2=1,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
| offset | offset | 表示对角线偏移量。  |
|  dim1           |  axis1         | 获取对角线的二维平面的第一维，仅参数名不一致。        |
|  dim2           |  axis2         | 获取对角线的二维平面的第二维，仅参数名不一致。        |
