## [ 仅参数名不一致 ]torch.ormqr

### [torch.ormqr](https://pytorch.org/docs/stable/generated/torch.ormqr.html#torch.ormqr)

```python
torch.ormqr(input, tau, other, left=True, transpose=False, *, out=None)
```

### [paddle.linalg.ormqr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/ormqr_cn.html#ormqr)

```python
paddle.linalg.ormqr(x, tau, other, left=True, transpose=False)
```

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                           |
| --------- | ------------ | ---------------------------------------------- |
| input     | x            | 输入的参数，用于表示矩阵 Q ，仅参数名字不一致 |
| tau       | tau          | Householder 反射系数，一致                     |
| other     | other        | 用于矩阵乘积，一致                             |
| left      | left         | 决定了矩阵乘积运算的顺序，一致                 |
| transpose | transpose    | 决定矩阵 Q 是否共轭转置变换，一致              |
| out       | -            | paddle 无此参数，需转写                        |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.ormqr(x, tau, other, left, transpose, out=y)

# Paddle 写法
paddle.assign(paddle.linalg.ormqr(x, tau, other, left, transpose) , y)
```
