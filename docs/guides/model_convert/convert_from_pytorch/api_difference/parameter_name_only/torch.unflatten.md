## [ 仅参数名不一致 ] torch.unflatten

### [torch.unflatten](https://pytorch.org/docs/stable/generated/torch.unflatten.html#torch.unflatten)

```python
torch.unflatten(input, dim, sizes)
```

### [paddle.unflatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/unflatten_cn.html#unflatten)

```python
paddle.unflatten(x, axis, shape, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入 Tensor，仅参数名不一致。                            |
| dim           | axis         | 需要变换的维度，仅参数名不一致。                          |
| sizes         | shape        | 维度变换的新形状，仅参数名不一致。                        |
