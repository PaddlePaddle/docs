## [ 仅参数名不一致 ] torch.select_scatter

### [torch.select_scatter](https://pytorch.org/docs/stable/generated/torch.select_scatter.html#torch-select-scatter)

```python
torch.select_scatter(input, src, dim, index)
```

### [paddle.select_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/select_scatter_cn.html)

```python
paddle.select_scatter(x, values, axis, index, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                     |
| ------- | ------------ | ---------------------------------------- |
| input   | x            | 输入张量，被嵌入的张量，仅参数名不一致。 |
| src     | values       | 用于嵌入的张量，仅参数名不一致。         |
| dim     | axis         | 嵌入的维度，仅参数名不一致。             |
| index   | index        | 选择的索引。                             |
