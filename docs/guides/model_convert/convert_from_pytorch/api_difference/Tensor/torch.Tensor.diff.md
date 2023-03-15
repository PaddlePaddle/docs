## [ 仅参数名不一致 ]torch.Tensor.diff

### [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html?highlight=diff#torch.diff)

```python
torch.diff(input, n=1, dim=- 1, prepend=None, append=None)
```

### [paddle.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diff_cn.html#diff)

```python
paddle.diff(x, n=1, axis=- 1, prepend=None, append=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                    | PaddlePaddle               | 备注                                                                                                                                                 |
| -------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| <center> input </center>   | <center> x </center>       | 待计算前向差值的输入 Tensor，仅参数名不同。                                                                                                          |
| <center> n </center>       | <center> n </center>       | 需要计算前向差值的次数，目前仅支持 n=1，默认值为 1。                                                                                                 |
| <center> dim </center>     | <center> axis </center>    | 沿着哪一维度计算前向差值，默认值为-1，也即最后一个维度，仅参数名不同。                                                                               |
| <center> prepend </center> | <center> prepend </center> | 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的前面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。 |
| <center> append </center>  | <center> append </center>  | 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的后面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。 |
