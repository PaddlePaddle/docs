## [torch 参数更多]torch.Tensor.quantile

### [torch.Tensor.quantile](https://pytorch.org/docs/1.13/generated/torch.Tensor.quantile.html#torch.Tensor.quantile)

```python
torch.Tensor.quantile(q, dim=None, keepdim=False, *, interpolation='linear')
```

### [paddle.Tensor.quantile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#quantile-q-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.quantile(q, axis=None, keepdim=False, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                     |
| ------- | ------------ | ------------------------ |
| q       | q            |待计算的分位数。          |
| dim     | axis         |指定对 x 进行计算的轴，仅参数名不一致。|
| keepdim | keepdim      |是否在输出 Tensor 中保留减小的维度。|
| interpolation | -      |两个数据点的插补取值方法，Paddle 无此功能，暂无转写方式。|
