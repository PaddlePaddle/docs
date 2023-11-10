## [参数不一致]torch.Tensor.cholesky_solve

### [torch.Tensor.cholesky_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.cholesky_solve.html#torch-tensor-cholesky-solve)

```python
torch.Tensor.cholesky_solve(input2,upper=False)
```

### [paddle.linalg.cholesky_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/cholesky_solve_cn.html#cholesky-solve)

```python
paddle.linalg.cholesky_solve(x,y,upper=False,name=None)
```

两者功能一致且参数用法一致，参数用法不一致，具体如下：：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                                       |
| ------- | ------------ | -------------------------------------------------------------------------- |
| -       | x            | Pytorch 中为当前 Tensor, 表示线性方程中的 B 矩阵。参数用法不一致，需要转写 |
| input2  | y            | 表示线性方程中 A 矩阵的 Cholesky 分解矩阵 u。仅参数名不一致                |
| upper   | upper        | 表示输入 x 是否是上三角矩阵，True 为上三角矩阵，False 为下三角矩阵。       |

### 转写示例
# torch 写法
x.cholesky_solve(y)

# paddle 写法
paddle.linalg.cholesky_solve(x, y)
```
