## [ 仅参数名不一致 ]torch.Tensor.diff

### [torch.Tensor.diff](https://pytorch.org/docs/stable/generated/torch.Tensor.diff.html?highlight=diff#torch.Tensor.diff)

```python
torch.Tensor.diff(n=1, dim=- 1, prepend=None, append=None)
```

### [paddle.Tensor.diff](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diff_cn.html#diff)

```python
paddle.Tensor.diff(n=1, axis=-1, prepend=None, append=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                                                                 |
| ------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| n       | n            | 需要计算前向差值的次数，目前仅支持 n=1，默认值为 1。                                                                                                 |
| dim     | axis         | 沿着哪一维度计算前向差值，默认值为-1，也即最后一个维度，仅参数名不同。                                                                               |
| prepend | prepend      | 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的前面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。 |
| append  | append       | 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的后面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。 |
