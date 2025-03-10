## [ 输入参数类型不一致 ]torch.linalg.lu_solve

### [torch.linalg.lu_solve](https://pytorch.org/docs/stable/generated/torch.linalg.lu_solve.html#torch.linalg.lu_solve)

```python
torch.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None)
```

### [paddle.linalg.lu_solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/lu_solve_cn.html)

```python
paddle.linalg.lu_solve(b, lu, pivots, trans="N", name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                  |
| ------- | ------------ | ----------------------------------------------------- |
| LU      | lu           | 表示 LU 分解结果矩阵，由 L、U 拼接组成，仅参数名不一致。  |
| pivots  | pivots       | 表示 LU 分解结果的主元信息 Tensor 。                    |
| B       | b            | 表示欲进行线性方程组求解的右值 Tensor ，仅参数名不一致。  |
| left    | -            | 表示待求解向量是否在左侧， Paddle 无此参数，需要转写。|
| adjoint | trans        | 表示是否使用转置 LU 分解结果， PyTorch 为 bool 类型，Paddle 为 str 类型，需要转写。|
| out     | -            | 表示输出的 Tensor 元组 ， Paddle 无此参数，需要转写。    |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.linalg.lu_solve(LU, pivots, B, out=A)

# Paddle 写法
y = paddle.linalg.lu_solve(B, LU, pivots)
paddle.assign(y, A)
```

#### left
```python
# PyTorch 写法  left=True
torch.linalg.lu_solve(LU, pivots, B, left=True)

# Paddle 写法
paddle.linalg.lu_solve(B, LU, pivots)

# PyTorch 写法  left=False
LU, pivots = torch.linalg.lu(A)
torch.linalg.lu_solve(LU, pivots, B, left=False)

# Paddle 写法
LU, pivots = paddle.linalg.lu(A.T)
paddle.linalg.lu_solve(B.T, LU, pivots).T
```

#### adjoint
```python
# PyTorch 写法  adjoint=False
torch.linalg.lu_solve(LU, pivots, B, adjoint=False)

# Paddle 写法
paddle.linalg.lu_solve(B, LU, pivots, trans="N")

# PyTorch 写法  adjoint=True
torch.linalg.lu_solve(LU, pivots, B, adjoint=True)

# Paddle 写法
paddle.linalg.lu_solve(B, LU, pivots, trans="C")
```
