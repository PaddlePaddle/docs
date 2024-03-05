## [ 仅参数名不一致 ]torch.Tensor.take_along_dim
### [torch.Tensor.take_along_dim](https://pytorch.org/docs/stable/generated/torch.Tensor.take_along_dim.html?highlight=torch+tensor+take_along_dim#torch.Tensor.take_along_dim)

```python
torch.Tensor.take_along_dim(indices,
                    dim)
```

### [paddle.Tensor.take_along_axis]( )

```python
paddle.Tensor.take_along_axis(indices,
                    axis,
                    broadcast=True)
```

两者功能一致，参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| indices         | indices         | 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。 |
| dim         | axis         |   指定沿着哪个维度获取对应的值，数据类型为：int，仅参数名不一致。 |
