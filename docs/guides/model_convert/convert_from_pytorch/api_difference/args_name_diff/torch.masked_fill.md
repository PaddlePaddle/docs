## [ 参数名兼容 ]torch.masked_fill
### [torch.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html#torch.Tensor.masked_fill)
```python
torch.masked_fill(input, mask, value)
```

### [paddle.masked_fill](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/masked_fill_cn.html#paddle.masked_fill)
```python
paddle.masked_fill(x, mask, value, name=None)
```

两者功能一致。Paddle 已兼容 PyTorch 风格关键字 `input`，因此：

- `paddle.masked_fill(input=..., mask=..., value=...)`
- `paddle.masked_fill(x=..., mask=..., value=...)`

两种写法等价，无需手动改名。

### 参数映射

| PyTorch | PaddlePaddle | 备注                                     |
| ------- | ------------ | ---------------------------------------- |
| input   | x / input    | 默认参数名是 `x`，同时兼容 `input`。      |
| mask    | mask         | 布尔张量，表示要填充的位置。               |
| value   | value        | 用于填充目标张量的值。                     |
