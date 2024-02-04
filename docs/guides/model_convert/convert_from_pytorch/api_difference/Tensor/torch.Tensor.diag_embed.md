## [ 参数完全一致 ]torch.Tensor.diag_embed

### [torch.Tensor.diag\_embed](https://pytorch.org/docs/stable/generated/torch.Tensor.diag_embed.html)

```python
torch.Tensor.diag_embed(offset=0, dim1=-2, dim2=-1)
```

### [paddle.Tensor.diag\_embed]()

```python
paddle.Tensor.diag_embed(offset=0, dim1=-2, dim2=-1, name=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| offset  | offset       | 从指定的二维平面中获取对角线的位置，默认值为 0，即主对角线。 |
| dim1    | dim1         | 填充对角线的二维平面的第一维，默认值为 -2。 |
| dim2    | dim2         | 填充对角线的二维平面的第二维，默认值为 -1。 |
