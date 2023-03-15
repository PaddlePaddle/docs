## [ 仅参数名不一致 ]torch.Tensor.cumsum

### [torch.cumsum](https://pytorch.org/docs/stable/generated/torch.cumsum.html?highlight=cumsum#torch.cumsum)

```python
torch.cumsum(input, dim, *, dtype=None, out=None)
```

### [paddle.cumsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cumsum_cn.html#cumsum)

```python
paddle.cumsum(x, axis=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle             | 备注                                      |
| ------------------------ | ------------------------ | ----------------------------------------- |
| <center> input </center> | <center> x </center>     | 需要进行累加操作的 Tensor，仅参数名不同。 |
| <center> dim </center>   | <center> axis </center>  | 需要累加的维度，仅参数名不同。            |
| <center> dtype </center> | <center> dtype </center> | 输出 Tensor 的数据类型。                  |
