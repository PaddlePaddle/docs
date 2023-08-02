## [仅参数名不一致]torch.cumulative_trapezoid

### [torch.cumulative_trapezoid](https://pytorch.org/docs/stable/generated/torch.cumulative_trapezoid.html#torch.cumulative_trapezoid)

```python
torch.cumulative_trapezoid(y, x=None, *, dx=None, dim=-1)
```

### [paddle.cumulative_trapezoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cumulative_trapezoid_cn.html)

```python
paddle.cumulative_trapezoid(y, x=None, dx=None, axis=-1, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                              |
| ------- | ------------ | ------------------------------------------------- |
| y       | y            | 输入多维 Tensor。                                 |
| x       | x            | y 中数值对应的浮点数所组成的 Tensor。             |
| dx      | dx           | 相邻采样点之间的常数间隔。                        |
| dim     | axis         | 计算 trapezoid rule 时 y 的维度，仅参数名不一致。 |
