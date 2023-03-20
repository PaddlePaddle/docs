## [ 仅参数名不一致 ] torch.Tensor.pow

### [torch.Tensor.pow](https://pytorch.org/docs/stable/generated/torch.Tensor.pow.html?highlight=pow#torch.Tensor.pow)

```python
torch.Tensor.pow(input, exponent, *, out=None)
```

### [paddle.Tensor.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html=)

```python
paddle.pow(x, y, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| input   | x            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
| vec2    | y            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
| out     | name         | 一般无需设置，默认值为 None。                      |
