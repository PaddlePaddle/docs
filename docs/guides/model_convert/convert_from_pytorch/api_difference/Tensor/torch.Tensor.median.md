## [ 仅参数名不一致 ]torch.Tensor.median

### [torch.Tensor.median](https://pytorch.org/docs/stable/generated/torch.Tensor.median.html)

```python
torch.Tensor.median(dim=None, keepdim=False)
```

### [paddle.Tensor.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#median-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.median(axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 指定对 x 进行计算的轴，仅参数名不一致。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
