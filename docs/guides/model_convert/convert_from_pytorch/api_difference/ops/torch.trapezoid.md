## [仅参数名不一致]torch.trapezoid

### [torch.trapezoid](https://pytorch.org/docs/stable/generated/torch.trapezoid.html#torch.trapezoid)

```python
torch.trapezoid(y, x=None, *, dx=None, dim=- 1)
```

### [paddle.trapezoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/trapezoid_cn.html#trapezoid)

```python
paddle.trapezoid(y, x=None, dx=None, axis=- 1, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                              |
| ------- | ------------ | ------------------------------------------------- |
| y       | y            | 输入多维 Tensor。                                 |
| x       | x            | y 中数值对应的浮点数所组成的 Tensor。             |
| dx      | dx           | 相邻采样点之间的常数间隔。                        |
| dim     | axis         | 计算 trapezoid rule 时 y 的维度，仅参数名不一致。 |
