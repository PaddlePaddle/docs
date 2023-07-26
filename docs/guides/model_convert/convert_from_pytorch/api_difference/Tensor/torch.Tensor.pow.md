## [ 仅参数名不一致 ] torch.Tensor.pow

### [torch.Tensor.pow](https://pytorch.org/docs/stable/generated/torch.Tensor.pow.html?highlight=pow#torch.Tensor.pow)

```python
torch.Tensor.pow(exponent)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html)

```python
paddle.pow(x, y)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                               |
| -------- | ------------ | -------------------------------------------------- |
| exponent | y            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
