## torch.diagonal
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
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| dim1         | axis1        | 获取对角线的二维平面的第一维。        |
| dim2         | axis2        | 获取对角线的二维平面的第二维。        |
