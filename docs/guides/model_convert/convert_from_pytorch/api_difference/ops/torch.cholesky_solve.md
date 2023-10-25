## [组合替代实现]torch.cholesky_solve
### [torch.cholesky_solve](https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html?highlight=cholesky_solve#torch.cholesky_solve)
```python
torch.cholesky_solve(input, input2, upper=False, out=None)
```

###  功能介绍
用于计算对称正定矩阵的逆矩阵，并与另一个矩阵相乘，公式为：
> 当`upper`为 False 时，
> $inv=(uu^T)^{-1}b$ ；
> 当`upper`为 True 时，
> $inv=(u^Tu)^{-1}b$ 。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

```python
import paddle

def cholesky_solve(input, input2, upper=False, out=None) :
    u = paddle.cholesky(input, False)
    ut = paddle.transpose(u, perm=[1, 0])
    if upper:
        out = paddle.inverse(paddle.matmul(ut, u))
    else:
        out = paddle.inverse(paddle.matmul(u, ut))
    out = paddle.matmul(out, input2)
    return out
```
