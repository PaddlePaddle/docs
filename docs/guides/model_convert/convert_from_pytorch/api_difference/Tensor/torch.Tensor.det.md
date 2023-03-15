## [ 仅参数名不一致 ]torch.Tensor.det

### [torch.linalg.det](https://pytorch.org/docs/stable/generated/torch.linalg.det.html#torch.linalg.det)

```python
torch.linalg.det(A, *, out=None)
```

### [paddle.linalg.det]()

```python
paddle.linalg.det(x)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch              | PaddlePaddle         | 备注                                                                                     |
| -------------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| <center> A </center> | <center> x </center> | 输入一个或批量矩阵。x 的形状应为 [*, M, M]，其中 \* 为零或更大的批次维度，仅参数名不同。 |
