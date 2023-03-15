## [ paddle 参数更多 ]torch.Tensor.logcumsumexp

### [torch.logcumsumexp](https://pytorch.org/docs/stable/generated/torch.logcumsumexp.html?highlight=logcumsumexp#torch.logcumsumexp)

```python
torch.logcumsumexp(input, dim, *, out=None)
```

### [paddle.logcumsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logcumsumexp_cn.html#logcumsumexp)

```python
paddle.logcumsumexp(x, axis=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，paddle 参数更多，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle             | 备注                                                                                                      |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------- |
| <center> input </center> | <center> x </center>     | 需要进行操作的 Tensor，仅参数名不同。                                                                     |
| <center> dim </center>   | <center> axis </center>  | 指明需要计算的维，仅参数名不同。 paddle 中默认 None。                                                     |
| <center> - </center>     | <center> dtype </center> | 输出 Tensor 的数据类型，支持 float32、float64。如果指定了，那么在执行操作之前，输入张量将被转换为 dtype。 |
