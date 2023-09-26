## [ 仅参数名不一致 ]torch.Tensor.cummax

### [torch.Tensor.cummax](https://pytorch.org/docs/stable/generated/torch.Tensor.cummax.html?highlight=cummax#torch.Tensor.cummax)

```python
torch.Tensor.cummax(dim, dtype=None)
```

### [paddle.Tensor.cummax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cummax_cn.html)

```python
paddle.Tensor.cummax(axis=None, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| dim     | axis         | 需要累加的维度，仅参数名不一致。 |
| dtype   | dtype        | 输出 Tensor 的数据类型。       |
