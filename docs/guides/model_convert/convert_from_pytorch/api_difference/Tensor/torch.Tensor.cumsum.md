## [ 仅参数名不一致 ]torch.Tensor.cumsum

### [torch.Tensor.cumsum](https://pytorch.org/docs/stable/generated/torch.Tensor.cumsum.html?highlight=cumsum#torch.Tensor.cumsum)

```python
torch.Tensor.cumsum(dim, dtype=None)
```

### [paddle.Tensor.cumsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cumsum-axis-none-dtype-none-name-none)

```python
paddle.Tensor.cumsum(axis=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| dim     | axis         | 需要累加的维度，仅参数名不一致。 |
| dtype   | dtype        | 输出 Tensor 的数据类型。       |
