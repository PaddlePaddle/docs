## [ 仅 API 调用方式不一致 ]torch.Tensor.ormqr
### [torch.Tensor.ormqr](https://pytorch.org/docs/stable/generated/torch.Tensor.orgqr.html#torch.Tensor.orgqr)
```python
torch.Tensor.ormqr(input2, input3, left=True, transpose=False)
```

### [paddle.linalg.ormqr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/ormqr_cn.html#ormqr)
```python
paddle.linalg.ormqr(x, tau, y, left=True, transpose=False)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
