## [ 仅参数名不一致 ]torch.Tensor.cumsum_
### [torch.Tensor.cumsum\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cumsum_.html#torch.Tensor.cumsum_)
```python
torch.Tensor.cumsum_(dim, *, dtype=None)
```

### [paddle.Tensor.cumsum_]()
```python
paddle.Tensor.cumsum_(axis=None, dtype=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| dim     | axis         | 需要累加的维度，仅参数名不一致。 |
| dtype   | dtype        | 输出 Tensor 的数据类型。       |
