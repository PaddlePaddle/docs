## [ 仅参数名不一致 ]torch.Tensor.isclose

### [torch.Tensor.isclose](https://pytorch.org/docs/stable/generated/torch.Tensor.isclose.html)

```python
torch.Tensor.isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False)
```

### [paddle.Tensor.isclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#isclose-x-y-rtol-1e-05-atol-1e-08-equal-nan-false-name-none)

```python
paddle.Tensor.isclose(y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注 |
| --------- | ------------ | -- |
| other     | y            | 输入的 Tensor，仅参数名不一致。 |
| rtol      | rtol         | 相对容忍误差。 |
| atol      | atol         | 绝对容忍误差。 |
| equal_nan | equal_nan    | 如果设置为 True，则两个 NaN 数值将被视为相等，默认值为 False。 |
