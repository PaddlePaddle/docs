## [ 仅参数名不一致 ]torch.Tensor.cumprod

### [torch.cumprod](https://pytorch.org/docs/stable/generated/torch.cumprod.html?highlight=cumprod#torch.cumprod)

```python
torch.cumprod(input, dim, *, dtype=None, out=None)
```

### [paddle.cumprod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cumprod_cn.html#cumprod)

```python
paddle.cumprod(x, dim=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle             | 备注                                                                                                                 |
| ------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| <center> input </center> | <center> x </center>     | 累乘的输入，需要进行累乘操作的 Tensor，仅参数名不同。                                                                |
| <center> dim </center>   | <center> dim </center>   | 指明需要累乘的维度。                                                                                                 |
| <center> dtype </center> | <center> dtype </center> | 返回张量所需的数据类型。dtype 如果指定，则在执行操作之前将输入张量转换为指定的 dtype，这对于防止数据类型溢出很有用。 |
