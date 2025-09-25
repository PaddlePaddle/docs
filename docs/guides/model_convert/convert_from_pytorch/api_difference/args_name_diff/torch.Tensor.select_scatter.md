## [ 仅参数名不一致 ] torch.Tensor.select_scatter

### [torch.Tensor.select_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.select_scatter.html#torch-select-scatter)

```python
torch.Tensor.select_scatter(src, dim, index)
```

### [paddle.Tensor.select_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#)

```python
paddle.Tensor.select_scatter(values, axis, index, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                     |
| ------- | ------------ | ---------------------------------------- |
| src     | values       | 用于嵌入的张量，仅参数名不一致。         |
| dim     | axis         | 嵌入的维度，仅参数名不一致。             |
| index   | index        | 选择的索引。                             |
