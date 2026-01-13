## [ 仅参数名不一致 ]torch.nn.utils.parametrizations.spectral_norm

### [torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html#torch.nn.utils.parametrizations.spectral_norm)

```python
torch.nn.utils.parametrizations.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None)
```

### [paddle.nn.utils.spectral_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/spectral_norm_cn.html#spectral-norm)

```python
paddle.nn.utils.spectral_norm(layer, name='weight', n_power_iterations=1, eps=1e-12, dim=None)
```

PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle       | 备注                                     |
| ------------------ | ------------------ | ---------------------------------------- |
| module             | layer              | 要添加权重谱归一化的层，仅参数名不一致。 |
| name               | name               | 权重参数的名字。                         |
| n_power_iterations | n_power_iterations | 将用于计算的 SpectralNorm 幂迭代次数。   |
| eps                | eps                | 用于保证计算中的数值稳定性。             |
| dim                | dim                | 重塑为矩阵之前应排列到第一个的维度索引。 |
