## [ 仅参数名不一致 ]torch.Tensor.diagflat

### [torch.diagflat](https://pytorch.org/docs/stable/generated/torch.diagflat.html?highlight=diagflat#torch.diagflat)

```python
torch.diagflat(input, offset=0)
```

### [paddle.diagflat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diagflat_cn.html#diagflat)

```python
paddle.diagflat(x, offset=0, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                   | PaddlePaddle              | 备注                                                               |
| ------------------------- | ------------------------- | ------------------------------------------------------------------ |
| <center> input </center>  | <center> x </center>      | 输入的 Tensor，仅参数名不同。                                      |
| <center> offset </center> | <center> offset </center> | 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。 |
