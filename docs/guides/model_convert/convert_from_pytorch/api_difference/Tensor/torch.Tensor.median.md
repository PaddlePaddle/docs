## [ paddle 参数更多 ]torch.Tensor.median

### [torch.Tensor.median](https://pytorch.org/docs/stable/generated/torch.Tensor.median.html)

```python
torch.Tensor.median(dim=None, keepdim=False)
```

### [paddle.Tensor.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#median-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.median(axis=None, keepdim=False, mode='avg', name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 指定对 x 进行计算的轴，仅参数名不一致。 |
| ------- | mode         | 当 x 在所需要计算的轴上有偶数个元素时，选择使用平均值或最小值确定中位数的值， PyTorch 无此参数， Paddle 需设置为 'min'。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
