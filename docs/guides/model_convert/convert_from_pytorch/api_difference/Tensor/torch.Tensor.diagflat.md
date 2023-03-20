## [ 仅 paddle 参数更多 ]torch.Tensor.diagflat

### [torch.Tensor.diagflat](https://pytorch.org/docs/stable/generated/torch.Tensor.diagflat.html?highlight=diagflat#torch.Tensor.diagflat)

```python
Tensor.diagflat(offset=0)
```

### [paddle.Tensor.diagflat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagflat_cn.html#diagflat)

```python
paddle.diagflat(x, offset=0, name=None)
```

两者功能一致，其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                               |
| ------- | ------------ | ------------------------------------------------------------------ |
| -       | x            | 输入的 Tensor，仅参数名不同。                                      |
| offset  | offset       | 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。 |
