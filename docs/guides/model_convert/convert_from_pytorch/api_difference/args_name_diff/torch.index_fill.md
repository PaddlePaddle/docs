## [ 仅参数名不一致 ]torch.index_fill
### [torch.index_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill.html#torch.Tensor.index_fill)
```python
torch.index_fill(input, dim, index, value)
```

### [paddle.index\_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/index_fill_cn.html#paddle.index_fill)
```python
paddle.index_fill(x, index, axis, value, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                               |
| ------- | ------------ | ---------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。    |
| dim     | axis         | 表示进行运算的轴，仅参数名不一致。 |
| index   | index        | 包含索引下标的 1-D Tensor。        |
| value   | value        | 填充的值。                         |
