## [仅参数名不一致]torch.cholesky_solve

### [torch.cholesky_solve](https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html?highlight=cholesky#torch.cholesky_solve)

```python
torch.cholesky_solve(input, input2, upper=False, *, out=None)
```

### [paddle.linalg.cholesky_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/cholesky_solve_cn.html#cholesky-solve)

```python
paddle.linalg.cholesky_solve(x, y, upper=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                                               |
| ------- | ------------ | ---------------------------------------------------------------------------------- |
| input   | x            | 表示线性方程中的 B 矩阵。仅参数名不一致                                            |
| input2  | y            | 表示线性方程中 A 矩阵的 Cholesky 分解矩阵 u。仅参数名不一致                        |
| upper   | upper        | 表示输入 x 是否是上三角矩阵，True 为上三角矩阵，False 为下三角矩阵。|
