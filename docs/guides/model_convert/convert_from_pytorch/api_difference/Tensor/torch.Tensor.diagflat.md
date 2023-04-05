## [ 参数完全一致 ]torch.Tensor.diagflat

### [torch.Tensor.diagflat](https://pytorch.org/docs/stable/generated/torch.Tensor.diagflat.html?highlight=diagflat#torch.Tensor.diagflat)

```python
torch.Tensor.diagflat(offset=0)
```

### [paddle.diagflat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagflat_cn.html#diagflat)

```python
paddle.diagflat(x, offset=0, name=None)
```

两者功能参数完全一致，其中 torch 是类成员函数，而 paddle 是非类成员函数，因此第一个参数 `x` 不进行对比，其他参数的映射为：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                               |
| ------- | ------------ | ------------------------------------------------------------------ |
| -       | x            | 输入的 Tensor，仅参数名不同。                                      |
| offset  | offset       | 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。 |
