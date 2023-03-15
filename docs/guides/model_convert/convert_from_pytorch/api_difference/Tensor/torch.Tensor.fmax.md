## [ 仅参数名不一致 ]torch.Tensor.fmax

### [torch.fmax](https://pytorch.org/docs/stable/generated/torch.fmax.html?highlight=fmax#torch.fmax)

```python
torch.fmax(input, other, *, out=None)
```

### [paddle.fmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fmax_cn.html#fmax)

```python
paddle.fmax(x, y, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                                |
| ------------------------ | -------------------- | ----------------------------------- |
| <center> input </center> | <center> x </center> | 输入的第一个 Tensor，仅参数名不同。 |
| <center> other </center> | <center> y </center> | 输入的第二个 Tensor，仅参数名不同。 |
