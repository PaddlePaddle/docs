## [ 仅参数名不一致 ]torch.Tensor.cholesky_solve

### [torch.Tensor.cholesky_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.cholesky_solve.html#torch-tensor-cholesky-solve)

```python
torch.Tensor.cholesky_solve(input2, upper=False)
```

### [paddle.Tensor.cholesky_solve]()

```python
paddle.Tensor.cholesky_solve(y, upper=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                       |
| ------- | ------------ | -------------------------------------------------------------------------- |
| input2  | y            | 表示线性方程中 A 矩阵的 Cholesky 分解矩阵 u。仅参数名不一致。              |
| upper   | upper        | 表示输入 x 是否是上三角矩阵，True 为上三角矩阵，False 为下三角矩阵。       |
