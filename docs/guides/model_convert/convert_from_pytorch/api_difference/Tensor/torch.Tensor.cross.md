## [ 仅参数名不一致 ]torch.Tensor.cross

### [torch.cross](https://pytorch.org/docs/stable/generated/torch.cross.html#torch.cross)

```python
torch.cross(input, other, dim=None, *, out=None)
```

### [paddle.cross](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cross_cn.html#cross)

```python
paddle.cross(x, y, axis=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle            | 备注                                   |
| ------------------------ | ----------------------- | -------------------------------------- |
| <center> input </center> | <center> x </center>    | 第一个输入 Tensor，仅参数名不同。      |
|  |
| <center> other </center> | <center> y </center>    | 第二个输入 Tensor，仅参数名不同。      |
| <center> dim </center>   | <center> axis </center> | 沿此维度进行向量积操作，仅参数名不同。 |
