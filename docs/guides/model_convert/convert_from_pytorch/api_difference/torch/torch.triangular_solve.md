## [返回参数类型不一致]torch.triangular_solve

### [torch.triangular_solve](https://pytorch.org/docs/stable/generated/torch.triangular_solve.html#torch.triangular_solve)

```python
torch.triangular_solve(input, A, upper=True, transpose=False, unitriangular=False, *, out=None)
```

### [paddle.linalg.triangular_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/triangular_solve_cn.html)

```python
paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                        |
| ------------- | ------------- | ----------------------------------------------------------- |
| input         | y             | 线性方程组左边的系数方阵，仅参数名不一致。                  |
| A             | x             | 线性方程组右边的矩阵，仅参数名不一致。                      |
| upper         | upper         | 对系数矩阵 x 取上三角还是下三角。                           |
| transpose     | transpose     | 是否对系数矩阵 x 进行转置。                                 |
| unitriangular | unitriangular | 如果为 True，则将系数矩阵 x 对角线元素假设为 1 来求解方程。 |
| out           | -             | 表示输出的 Tensor，Paddle 无此参数，需要转写。          |
| 返回值         | 返回值         | Pytorch 返回两个 Tensor：solution 与 A，Paddle 仅返回一个 Tensor：solution，需要转写。          |

### 转写示例
#### 返回值
```python
# PyTorch 写法:
torch.triangular_solve(b, A)

# Paddle 写法:
## 注：Paddle 将 A 与 b 交换
tuple(paddle.linalg.triangular_solve(A, b), A)
```

#### out 参数：输出的 Tensor
```python
# PyTorch 写法:
torch.triangular_solve(b, A, out=(y1, y2))

# Paddle 写法:
paddle.assign(paddle.linalg.triangular_solve(A, b), y1)
paddle.assign(A, y2)
```
