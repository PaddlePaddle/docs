## [ 仅参数名不一致 ]torch.diagonal
### [torch.diagonal](https://pytorch.org/docs/stable/generated/torch.diagonal.html?highlight=diagonal#torch.diagonal)

```python
torch.diagonal(input,
               offset=0,
               dim1=0,
               dim2=1))
```

### [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diagonal_cn.html#diagonal)

```python
paddle.diagonal(x,
                offset=0,
                axis1=0,
                axis2=1,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| offset | offset | 表示对角线偏移量。  |
| <font color='red'> dim1 </font>          | <font color='red'> axis1 </font>        | 获取对角线的二维平面的第一维，仅参数名不一致。        |
| <font color='red'> dim2 </font>          | <font color='red'> axis2 </font>        | 获取对角线的二维平面的第二维，仅参数名不一致。        |
