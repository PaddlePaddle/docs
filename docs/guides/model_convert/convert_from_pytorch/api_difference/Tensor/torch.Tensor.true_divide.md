# [仅参数名不一致]torch.Tensor.true_divide

[torch.Tensor.true_divide](https://pytorch.org/docs/stable/generated/torch.true_divide.html#torch-true-divide)

```python
torch.true_divide(dividend, divisor, *, out)
```

[paddle.Tensor.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/divide_cn.html#divide)

```python
paddle.divide(x, y, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

| PyTorch  | PaddlePaddle |               备注               |
| -------- | :----------: | :------------------------------: |
| dividend |      x       | 输入的第一个 Tensor，仅参数名不同 |
| divisor  |      y       | 输入的第二个 Tensor，仅参数名不同 |
