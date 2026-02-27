## [ 仅参数名不一致 ]torch.diagonal_scatter
### [torch.diagonal\_scatter](https://docs.pytorch.org/docs/stable/generated/torch.diagonal_scatter.html#torch.diagonal_scatter)
```python
torch.diagonal_scatter(input,
                       src,
                       offset=0,
                       dim1=0,
                       dim2=1)
```

### [paddle.diagonal\_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_scatter_cn.html#paddle.diagonal_scatter)
```python
paddle.diagonal_scatter(x,
                        y,
                        offset=0,
                        axis1=0,
                        axis2=1)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
|---------|--------------| -------------------------------------------------- |
|  input      |  x           | 输入张量，被嵌入的张量，仅参数名不一致。    |
|  src      |  y           | 用于嵌入的张量，仅参数名不一致。    |
|  offset      |  offset           | 从指定的二维平面嵌入对角线的位置，默认值为 0，即主对角线。    |
|  dim1      |  axis1           | 对角线的第一个维度，默认值为 0，仅参数名不一致。    |
|  dim2      |  axis2           | 对角线的第二个维度，默认值为 1，仅参数名不一致。    |
