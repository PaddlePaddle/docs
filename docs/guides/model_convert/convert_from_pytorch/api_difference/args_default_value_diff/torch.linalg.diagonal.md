## [ 参数默认值不一致 ]torch.linalg.diagonal
### [torch.linalg.diagonal](https://docs.pytorch.org/docs/stable/generated/torch.linalg.diagonal.html#torch.linalg.diagonal)
```python
torch.linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1)
```

### [paddle.diagonal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagonal_cn.html#paddle.diagonal)
```python
paddle.diagonal(x,
                offset=0,
                axis1=0,
                axis2=1,
                name=None)
```

两者功能一致且参数用法一致，参数默认值不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  A  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
| offset | offset | 表示对角线偏移量。  |
|  dim1           |  axis1         | 获取对角线的二维平面的第一维，参数默认值不一致。PyTorch 默认为`-2`，Paddle 默认为`0`，Paddle 需设置为与 PyTorch 一致。        |
|  dim2           |  axis2         | 获取对角线的二维平面的第二维，参数默认值不一致。PyTorch 默认为`-1`，Paddle 默认为`1`，Paddle 需设置为与 PyTorch 一致。        |
