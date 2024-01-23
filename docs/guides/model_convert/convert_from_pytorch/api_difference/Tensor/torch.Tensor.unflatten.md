## [ 仅参数名不一致 ] torch.Tensor.unflatten

### [torch.Tensor.unflatten](https://pytorch.org/docs/stable/generated/torch.Tensor.unflatten.html#torch.Tensor.unflatten)

```python
torch.Tensor.unflatten(dim, sizes)
```

### [paddle.Tensor.unflatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#unflatten-axis-shape-name-none)

```python
paddle.Tensor.unflatten(axis, shape, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 需要变换的维度，仅参数名不一致。                          |
| sizes         | shape        | 维度变换的新形状，仅参数名不一致。                        |
