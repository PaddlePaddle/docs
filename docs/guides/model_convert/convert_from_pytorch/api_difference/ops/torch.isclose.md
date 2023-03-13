## [ 仅参数名不一致 ]torch.isclose
### [torch.isclose](https://pytorch.org/docs/stable/generated/torch.isclose.html?highlight=isclose#torch.isclose)

```python
torch.isclose(input,
              other,
              rtol=1e-05,
              atol=1e-08,
              equal_nan=False)
```

### [paddle.isclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/isclose_cn.html#isclose)

```python
paddle.isclose(x,
               y,
               rtol=1e-05,
               atol=1e-08,
               equal_nan=False,
               name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| rtol | rtol | 表示相对容忍误差。 |
| atol | atol | 表示绝对容忍误差。 |
| equal_nan | equal_nan | 表示是否将两个 NaN 数值视为相等。 |
