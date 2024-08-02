## [ 仅参数名不一致 ] torch.masked_fill

### [torch.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html#torch.Tensor.masked_fill)

```python
torch.masked_fill(input, mask, value)
```

### [paddle.masked_fill](https://github.com/PaddlePaddle/Paddle/blob/1e3761d119643af19cb6f8a031a77f315d782409/python/paddle/tensor/manipulation.py#L5111)

```python
paddle.masked_fill(x, mask, value, name=None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                            |
| ------- | ------------ | ------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。 |
| mask    | mask         | 布尔张量，表示要填充的位置      |
| value   | value        | 用于填充目标张量的值            |
