## [仅参数名不一致]torch.Tensor.var

### [torch.Tensor.var](https://pytorch.org/docs/1.13/generated/torch.Tensor.var.html#torch.Tensor.var)

```python
torch.Tensor.var(dim, unbiased=True, keepdim=False)
```

### [paddle.Tensor.var](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#var-axis-none-unbiased-true-keepdim-false-name-none)

```python
paddle.Tensor.var(axis=None, unbiased=True, keepdim=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  dim |  axis  | 指定对 Tensor 进行计算的轴 ，仅参数名不一致。   |
| unbiased | unbiased | 表示是否使用无偏估计来计算方差。 |
| keepdim | keepdim | 表示是否保留计算后的维度。 |
