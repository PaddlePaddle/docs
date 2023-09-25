## [torch 参数更多]torch.linalg.solve

### [torch.linalg.solve](https://pytorch.org/docs/stable/generated/torch.linalg.solve.html#torch.linalg.solve)

```python
torch.linalg.solve(A, B, *, left=True, out=None)
```

### [paddle.linalg.solve](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/solve_cn.html)

```python
paddle.linalg.solve(x, y, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | ------------------------------------------------------------ |
| A       | x            | 输入线性方程组求解的一个或一批方阵，仅参数名不一致。 |
| B       | y            | 输入线性方程组求解的右值，仅参数名不一致。           |
| left    | -            | 是否求解 AX = B 或 XA = B，Paddle 无此参数，暂无转写方式。   |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。           |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.solve(x1, x2, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.solve(x1, x2), y)
```
