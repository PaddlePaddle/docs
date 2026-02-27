## [ 仅参数名不一致 ]torch.Tensor.rot90
### [torch.Tensor.rot90](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.rot90.html#torch.Tensor.rot90)
```python
torch.Tensor.rot90(k, dims)
```

### [paddle.Tensor.rot90](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#rot90-k-1-axis-0-1-name-none)
```python
paddle.Tensor.rot90(k=1, axes=[0, 1])
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| k           | k         | 旋转方向和次数。 |
| dims    | axes    | 指定旋转的平面，仅参数名不一致。     |
