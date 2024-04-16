## [ 仅参数名不一致 ]torch.Tensor.renorm

### [torch.Tensor.renorm](https://pytorch.org/docs/stable/generated/torch.Tensor.renorm.html#torch-tensor-renorm)

```python
torch.Tensor.renorm(p, dim, maxnorm)
```

### [paddle.Tensor.renorm]()

```python
paddle.Tensor.renorm(p, axis, max_norm)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| p       | p            | 表示 p-范数计算的 p 值。|
| dim     | axis         | 表示切分的维度，仅参数名不一致。                                    |
| maxnorm | max_norm     | 表示子张量的 p-范数最大值，仅参数名不一致。          |
