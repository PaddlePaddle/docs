## [ 参数不一致 ]torch.Tensor.lstsq

### [torch.Tensor.lstsq](https://pytorch.org/docs/1.9.0/generated/torch.Tensor.lstsq.html?highlight=torch%20tensor%20lstsq#torch.Tensor.lstsq)

```python
torch.Tensor.lstsq(A)
```

### [paddle.Tensor.lstsq]()

```python
paddle.Tensor.lstsq(y, rcond=None, driver=None, name=None)
```

两者功能一致，参数不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                            |
| ------- | ------------ | ----------------------------------------------------------------------------------------------- |
| A       | -            | 线性方程组系数矩阵，Paddle 需要转写。                                                           |
| -       | y            | 线性方程组右边的矩阵，Paddle 需要转写。                                                         |
| -       | rcond        | 用来决定 x 有效秩的 float 型浮点数。PyTorch 无此参数，Paddle 保持默认即可。                     |
| -       | driver       | 用来指定计算使用的 LAPACK 库方法。PyTorch 无此参数，Paddle 保持默认即可。                       |
| 返回值  | 返回值       | PyTorch 返回 solution、QR ，Paddle 返回 solution、residuals、rank、 singular_values，Paddle 与 PyTorch 仅第一个返回值相同，其他返回值结果不同，暂无转写方式。 |

### 转写示例

#### A 参数转写

```python
# PyTorch 写法:
A = torch.tensor([[1, 1, 1],
                  [0, 2, 1],
                  [0, 0,-1]], dtype=torch.float64)
y = torch.tensor([[0], [-9], [5]], dtype=torch.float64)
y.lstsq(A)

# Paddle 写法:
A = paddle.to_tensor([[1, 1, 1],
                      [0, 2, 1],
                      [0, 0,-1]], dtype=paddle.float64)
y = paddle.to_tensor([[0], [-9], [5]], dtype=paddle.float64)
A.lstsq(y)
```
