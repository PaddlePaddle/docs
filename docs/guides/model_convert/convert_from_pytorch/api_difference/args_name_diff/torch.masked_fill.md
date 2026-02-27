## [ 仅参数名不一致 ]torch.masked_fill
### [torch.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html#torch.Tensor.masked_fill)
```python
torch.masked_fill(input, mask, value)
```

### [paddle.masked\_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/masked_fill_cn.html#paddle.masked_fill)
```python
paddle.masked_fill(x, mask, value, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                            |
| ------- | ------------ | ------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。 |
| mask    | mask         | 布尔张量，表示要填充的位置。      |
| value   | value        | 用于填充目标张量的值。            |
