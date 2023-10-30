## [ 参数完全一致 ] torch.diag_embed

### [torch.diag_embed](https://pytorch.org/docs/stable/generated/torch.diag_embed.html)

```python
torch.diag_embed(input, offset=0, dim1=-2, dim2=-1)
```

### [paddle.diag_embed](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diag_embed_cn.html)

```python
paddle.diag_embed(input, offset=0, dim1=- 2, dim2=- 1)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| input       | input        | 输入变量，至少为 1D 数组，支持数据类型为 float32、float64、int32、int64。                   |
| offset      | offset       | 从指定的二维平面中获取对角线的位置，默认值为 0，即主对角线。                                 |
| dim1        | dim1         | 填充对角线的二维平面的第一维，默认值为 -2。                                                 |
| dim2        | dim2         | 填充对角线的二维平面的第二维，默认值为 -1。                                                 |
