## [ 仅参数名不一致 ]torch.Tensor.quantile

### [torch.Tensor.quantile](https://pytorch.org/docs/stable/generated/torch.Tensor.quantile.html#torch.Tensor.quantile)

```python
torch.Tensor.quantile(q, dim=None, keepdim=False, *, interpolation='linear')
```

### [paddle.Tensor.quantile](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#quantile-q-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.quantile(q, axis=None, keepdim=False, interpolation='linear', name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                     |
| ------- | ------------ | ------------------------ |
| q       | q            | 待计算的分位数。          |
| dim     | axis         | 指定对 x 进行计算的轴，仅参数名不一致。|
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。|
| interpolation | interpolation | 两个数据点的插补取值方法。 |
