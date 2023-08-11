## [torch 参数更多]torch.triangular_solve

### [torch.triangular_solve](https://pytorch.org/docs/stable/generated/torch.triangular_solve.html#torch.triangular_solve)

```python
torch.triangular_solve(b, A, upper=True, transpose=False, unitriangular=False, *, out=None)
```

### [paddle.linalg.triangular_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/triangular_solve_cn.html)

```python
paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                        |
| ------------- | ------------- | ----------------------------------------------------------- |
| A             | x             | 线性方程组左边的系数方阵，仅参数名不一致。                  |
| b             | y             | 线性方程组右边的矩阵，仅参数名不一致。                      |
| upper         | upper         | 对系数矩阵 x 取上三角还是下三角。                           |
| transpose     | transpose     | 是否对系数矩阵 x 进行转置。                                 |
| unitriangular | unitriangular | 如果为 True，则将系数矩阵 x 对角线元素假设为 1 来求解方程。 |
| out           | -             | 表示输出的 Tensor，Paddle 无此参数，需要转写。          |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.triangular_solve(x2, x1, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.triangular_solve(x1, x2), y)
```
