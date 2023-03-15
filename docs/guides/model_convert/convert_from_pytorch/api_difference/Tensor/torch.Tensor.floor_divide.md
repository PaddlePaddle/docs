## [ 仅参数名不一致 ]torch.Tensor.floor_divide

### [torch.floor_divide](https://pytorch.org/docs/stable/generated/torch.floor_divide.html?highlight=floor_divide#torch.floor_divide)

```python
torch.floor_divide(input, other, *, out=None)
```

### [paddle.floor_divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_divide_cn.html#floor-divide)

```python
paddle.floor_divide(x, y, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                        |
| ------------------------ | -------------------- | --------------------------- |
| <center> input </center> | <center> x </center> | 多维 Tensor，仅参数名不同。 |
| <center> other </center> | <center> y </center> | 多维 Tensor，仅参数名不同。 |
