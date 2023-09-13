## [ 仅参数名不一致 ]torch.Tensor.addmm_

### [torch.Tensor.addmm_](https://pytorch.org/docs/stable/generated/torch.Tensor.addmm_.html)

```python
torch.Tensor.addmm(mat1, mat2, *, beta=1, alpha=1)
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
