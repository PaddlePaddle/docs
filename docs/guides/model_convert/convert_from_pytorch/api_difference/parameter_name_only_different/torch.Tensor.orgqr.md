## [ 仅参数名不一致 ]torch.Tensor.orgqr

### [torch.Tensor.orgqr](https://pytorch.org/docs/stable/generated/torch.Tensor.orgqr.html#torch.Tensor.orgqr)

```python
torch.Tensor.orgqr(input2)
```

### [paddle.Tensor.householder_product](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/householder_product_cn.html#householder-product)

```python
paddle.Tensor.householder_product(tau, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| input2  | tau          | 用于计算矩阵乘积，仅参数名不同 |
