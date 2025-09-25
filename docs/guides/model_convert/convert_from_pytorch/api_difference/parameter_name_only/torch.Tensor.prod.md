## [ 仅参数名不一致 ] torch.Tensor.prod

### [torch.Tensor.prod](https://pytorch.org/docs/stable/generated/torch.prod.html#torch.prod)

```python
torch.Tensor.prod(dim, keepdim=False, *, dtype=None)
```

### [paddle.Tensor.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/prod_cn.html)

```python
paddle.Tensor.prod(axis=None, keepdim=False, dtype=None, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                          |
| ------- | ------------ | ------------------------------------------------------------- |
| dim     | axis         | 求乘积运算的维度，仅参数名不一致。                            |
| keepdim | keepdim      | 是否在输出 Tensor 中保留输入的维度，仅参数名不一致。          |
| dtype   | dtype        | 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。 |
| -       | name         | 一般无需设置，默认值为 None。                                 |
