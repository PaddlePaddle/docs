## [ 仅 paddle 参数更多 ] torch.Tensor.pow

### [torch.Tensor.pow](https://pytorch.org/docs/stable/generated/torch.Tensor.pow.html?highlight=pow#torch.Tensor.pow)

```python
torch.Tensor.pow(exponent)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html)

```python
paddle.pow(x, y)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                               |
| -------- | ------------ | -------------------------------------------------- |
| - | x            | 多维 Tensor，数据类型为 float16 、 float32 、 float64 、 int32 或 int64 。 |
| exponent | y            | 一个 N 维 Tensor 或者标量 Tensor，仅参数名不一致。 |
