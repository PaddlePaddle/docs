## [ 仅参数名不一致 ]torch.Tensor.index_fill
### [torch.Tensor.index_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill.html?highlight=index_fill#torch.Tensor.index_fill)
```python
torch.Tensor.index_fill(dim, index, value)
```

### [paddle.Tensor.index_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_fill_cn.html#index-fill)
```python
paddle.Tensor.index_fill(index, axis, value, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  dim  |  axis  | 表示进行运算的轴，仅参数名不一致。  |
|  index  |  index  | 包含索引下标的 1-D Tensor。  |
|  value  |  value  | 填充的值。  |
