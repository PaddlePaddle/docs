## [ torch 参数更多 ]torch.linalg.solve_triangular

### [torch.linalg.solve_triangular](https://pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html?highlight=torch+linalg+solve_triangular#torch.linalg.solve_triangular)

```python
# Pytorch 文档有误，测试第一个参数为 input
torch.linalg.solve_triangular(input, B, *, upper, left=True, unitriangular=False, out=None)
```

### [paddle.linalg.triangular_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/triangular_solve_cn.html)

```python
paddle.linalg.triangular_solve(x, y, upper=True, transpose=False, unitriangular=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle  | 备注                                                                                                              |
| ------------- | ------------- | ----------------------------------------------------------------------------------------------------------------- |
| input         | x             | 表示线性方程组左边的系数 Tensor ，仅参数名不一致。                                                                |
| B             | y             | 表示线性方程组右边的 Tensor ，仅参数名不一致。                                                                    |
| upper         | upper         | 表示对系数 Tensor 取上三角还是下三角。                                                                            |
| left          | transpose     | 表示是否对系数 Tensor 进行转置 ，PyTorch 和 Paddle 取值相反，PyTorch 默认为 True，Paddle 默认为 False，需要转写。 |
| unitriangular | unitriangular | 表示是否将系数 Tensor 对角线元素假设为 1 来求解方程。                                                             |
| out           | -             | 表示输出的 Tensor ， Paddle 无此参数，需要转写。                                                                  |

### 转写示例

#### left：表示系数 Tensor 的位置设置

```python
# Pytorch 写法, left 为 True
torch.linalg.solve_triangular(input, B, upper, left=True)

# Paddle 写法
paddle.linalg.triangular_solve(input, B, upper, transpose=False)

# Pytorch 写法, left 为 False
torch.linalg.solve_triangular(input, B, upper, left=False)

# Paddle 写法
paddle.linalg.triangular_solve(input, B, upper, transpose=True)
```

#### out：指定输出

```python
# Pytorch 写法
torch.linalg.solve_triangular(input, B, upper, left, unitriangular, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.triangular_solve(input, B, upper, transpose, unitriangular) , y)
```
