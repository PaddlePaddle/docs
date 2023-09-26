## [ 仅参数名不一致 ]torch.Tensor.cummin

### [torch.Tensor.cummin](https://pytorch.org/docs/stable/generated/torch.Tensor.cummin.html?highlight=cummin#torch.Tensor.cummin)

```python
torch.Tensor.cummin(dim, dtype=None)
```

### [paddle.Tensor.cummin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cummin_cn.html#cummin)

```python
paddle.Tensor.cummin(axis=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| dim     | axis         | 需要累加的维度，仅参数名不一致。 |
| dtype   | dtype        | 输出 Tensor 的数据类型。       |
