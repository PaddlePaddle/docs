## [ 仅参数名不一致 ]torch.nn.Unflatten
### [torch.nn.Unflatten](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Unflatten.html)
```python
torch.nn.Unflatten(dim, unflattened_size)
```

### [paddle.nn.Unflatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Unflatten_cn.html#paddle.nn.Unflatten)
```python
paddle.nn.Unflatten(axis, shape, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。                          |
| unflattened_size           | shape        | 需要展开的形状，仅参数名不一致。                          |
