## [ 仅参数名不一致 ]torch.Tensor.index_fill_
### [torch.Tensor.index_fill_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill_.html?highlight=index_fill_#torch.Tensor.index_fill_)
```python
torch.Tensor.index_fill_(dim, index, value)
```

### [paddle.Tensor.index_fill_]()
```python
paddle.Tensor.index_fill_(index, axis, value, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  dim  |  axis  | 表示进行运算的轴，仅参数名不一致。  |
|  index  |  index  | 包含索引下标的 1-D Tensor。  |
|  value  |  value  | 填充的值。  |
