## [ 仅参数名不一致 ]torch.Tensor.ormqr

### [torch.Tensor.ormqr](https://pytorch.org/docs/stable/generated/torch.Tensor.orgqr.html#torch.Tensor.orgqr)

```python
torch.Tensor.ormqr(input2, input3, left=True, transpose=False)
```

### [paddle.Tensor.ormqr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#tensor)

```python
paddle.Tensor.ormqr(tau, other, left=True, transpose=False)
```

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input2         | tau         | 仅参数名字不一致 |
| input3     | other         | 仅参数名字不一致 |
| left     | left         | 一致 |
| transpose     | transpose         | 一致 |
