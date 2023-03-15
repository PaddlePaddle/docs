## [ 仅参数名不一致 ]torch.Tensor.erf

### [torch.special.erf](https://pytorch.org/docs/stable/special.html#torch.special.erf)

```python
torch.special.erf(input, *, out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/erf_cn.html#erf)

```python
paddle.erf(x, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                              |
| ------------------------ | -------------------- | --------------------------------- |
| <center> input </center> | <center> x </center> | 输入的多维 Tensor，仅参数名不同。 |
