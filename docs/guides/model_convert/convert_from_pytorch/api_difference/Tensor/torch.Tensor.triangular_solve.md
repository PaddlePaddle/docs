## [参数不一致]torch.Tensor.triangular_solve

### [torch.Tensor.triangular_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.triangular_solve.html#torch.Tensor.triangular_solve)

```python
torch.Tensor.triangular_solve(A, upper=True, transpose=False, unitriangular=False)
```

### [paddle.Tensor.triangular_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#triangular-solve-b-upper-true-transpose-false-unitriangular-false-name-none)

```python
paddle.Tensor.triangular_solve(b, upper=True, transpose=False, unitriangular=False, name=None)
```

其中两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                        |
| ------------- | ------------- | ----------------------------------------------------------- |
| A             | -             | 线性方程组系数矩阵，Paddle 需要转写。                   |
| -             | b             | 线性方程组右边的矩阵，Paddle 需要转写。                 |
| upper         | upper         | 对系数矩阵 x 取上三角还是下三角。                           |
| transpose     | transpose     | 是否对系数矩阵 x 进行转置。                                 |
| unitriangular | unitriangular | 如果为 True，则将系数矩阵 x 对角线元素假设为 1 来求解方程。 |

### 转写示例

#### A：线性方程组系数矩阵

```python
# PyTorch 写法
A = torch.tensor([[1, 1, 1],
                  [0, 2, 1],
                  [0, 0,-1]], dtype=torch.float64)
b = torch.tensor([[0], [-9], [5]], dtype=torch.float64)
b.triangular_solve(A)

# Paddle 写法
A = paddle.to_tensor([[1, 1, 1],
                      [0, 2, 1],
                      [0, 0,-1]], dtype=paddle.float64)
b = paddle.to_tensor([[0], [-9], [5]], dtype=paddle.float64)
A.triangular_solve(b)

# 注：Paddle 将 A 与 b 交换
```
