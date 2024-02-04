## [ 仅参数名不一致 ]torch.Tensor.nanmedian

### [torch.Tensor.nanmedian](https://pytorch.org/docs/stable/generated/torch.Tensor.nanmedian.html)

```python
torch.Tensor.nanmedian(dim=None, keepdim=False)
```

### [paddle.Tensor.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#nanmedian-axis-none-keepdim-true-name-none)

```python
paddle.Tensor.nanmedian(axis=None, keepdim=True, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名与参数默认值不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 指定对 x 进行计算的轴。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度，PyTorch 默认值为 False，Paddle 默认值为 True。 |
