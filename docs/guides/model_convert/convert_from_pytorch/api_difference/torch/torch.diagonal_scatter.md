## [ 仅参数名不一致 ] torch.diagonal_scatter

### [torch.diagonal_scatter](https://pytorch.org/docs/stable/generated/torch.diagonal_scatter.html?highlight=diagonal_scatter#torch.diagonal_scatter)

```python
torch.diagonal_scatter(input,
                       src,
                       offset=0,
                       dim1=0,
                       dim2=1)
```

### [paddle.diagonal_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_scatter_cn.html)

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
| <font color='red'> input </font>     | <font color='red'> x </font>          | 输入张量，被嵌入的张量，仅参数名不一致。    |
| <font color='red'> src </font>     | <font color='red'> y </font>          | 用于嵌入的张量，仅参数名不一致。    |
| <font color='red'> offset </font>     | <font color='red'> offset </font>          | 从指定的二维平面嵌入对角线的位置，默认值为 0，即主对角线。    |
| <font color='red'> dim1 </font>     | <font color='red'> axis1 </font>          | 对角线的第一个维度，默认值为 0，仅参数名不一致。    |
| <font color='red'> dim2 </font>     | <font color='red'> axis2 </font>          | 对角线的第二个维度，默认值为 1，仅参数名不一致。    |
