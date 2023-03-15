## [ 仅参数名不一致 ]torch.Tensor.erfinv\_

### [torch.erfinv\_](https://pytorch.org/docs/stable/generated/torch.Tensor.erfinv_.html?highlight=erfinv_)

```python
Tensor.erfinv_(input)
```

### [paddle.erfinv\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/erfinv__cn.html#erfinv)

```python
paddle.erfinv_(x)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                                                               |
| ------------------------ | -------------------- | ------------------------------------------------------------------ |
| <center> input </center> | <center> x </center> | Inplace 版本的 erfinv API，对输入采用 Inplace 策略，仅参数名不同。 |
