## [返回参数类型不一致]torch.Tensor.triangular_solve

### [torch.Tensor.triangular_solve](https://pytorch.org/docs/stable/generated/torch.Tensor.triangular_solve.html#torch.Tensor.triangular_solve)

```python
torch.Tensor.triangular_solve(A, upper=True, transpose=False, unitriangular=False)
```

### [paddle.linalg.triangular_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/triangular_solve_cn.html)

```python
paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)
```

Pytorch 为 Tensor 类方法，Paddle 为普通函数，另外两者的返回 Tensor 个数不同。参数对应关系如下表所示：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                        |
| ------------- | ------------- | ----------------------------------------------------------- |
| A             | x             | 线性方程组系数矩阵。                                             |
| self          | y             | 线性方程组右边的矩阵，将调用 torch.Tensor 类方法的 self Tensor 传入。 |
| upper         | upper         | 对系数矩阵 x 取上三角还是下三角。                                |
| transpose     | transpose     | 是否对系数矩阵 x 进行转置。                                     |
| unitriangular | unitriangular | 如果为 True，则将系数矩阵 x 对角线元素假设为 1 来求解方程。         |
| 返回值         | 返回值         | Pytorch 返回两个 Tensor：solution 与 A，Paddle 仅返回一个 Tensor：solution。需要转写。  |

### 转写示例

#### 返回值
```python
# PyTorch 写法:
b.triangular_solve(A)

# Paddle 写法:
## 注：Paddle 将 A 与 b 交换
tuple(paddle.linalg.triangular_solve(A, b), A)
```
