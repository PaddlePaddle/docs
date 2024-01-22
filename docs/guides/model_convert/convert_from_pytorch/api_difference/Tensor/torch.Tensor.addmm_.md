## [ 仅参数名不一致 ]torch.Tensor.addmm_

### [torch.Tensor.addmm_](https://pytorch.org/docs/stable/generated/torch.Tensor.addmm_.html)

```python
torch.Tensor.addmm_(mat1, mat2, *, beta=1, alpha=1)
```

### [paddle.Tensor.addmm_]()

```python
paddle.Tensor.addmm_(x, y, alpha=1.0, beta=1.0)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle |               备注               |
| :------: | :----------: | :------------------------------: |
| mat1 |      x       | 表示输入的 Tensor，仅参数名不一致。 |
| mat2 |      y       | 表示输入的 Tensor，仅参数名不一致。 |
| alpha |   alpha     | 乘以 x*y 的标量，数据类型支持 float32、float64，默认值为 1.0。|
| beta  |    beta     | 乘以 input 的标量，数据类型支持 float32、float64，默认值为 1.0。|
