## [ 仅参数名不一致 ]torch.Tensor.ormqr

### [torch.Tensor.ormqr](https://pytorch.org/docs/stable/generated/torch.Tensor.orgqr.html#torch.Tensor.orgqr)

```python
torch.Tensor.ormqr(input2, input3, left=True, transpose=False)
```

### [paddle.linalg.ormqr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/ormqr_cn.html#ormqr)

```python
paddle.linalg.ormqr(x, tau, other, left=True, transpose=False)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                               |
| --------- | ------------ | ---------------------------------- |
| input2    | tau          | Householder 反射系数，仅参数名不同 |
| input3    | other        | 用于矩阵乘积，仅参数名不同         |
| left      | left         | 决定了矩阵乘积运算的顺序，一致     |
| transpose | transpose    | 决定矩阵 Q 是否共轭转置变换，一致  |
