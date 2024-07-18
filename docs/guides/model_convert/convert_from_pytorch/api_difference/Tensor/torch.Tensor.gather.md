## [ 仅参数名不一致 ]torch.Tensor.gather

### [torch.Tensor.gather](https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html?highlight=gather#torch.Tensor.gather)

```python
torch.Tensor.gather(dim, index)
```

### [paddle.Tensor.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#take-along-axis-arr-indices-axis-broadcast-true)

```python
paddle.Tensor.take_along_axis(indices, axis, broadcast=True)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                      |
| ------- | ------------ | ----------------------------------------- |
| dim     | axis         | 指定 index 获取输入的维度，仅参数名不一致。 |
| index   | indices      | 索引 Tensor，仅参数名不一致。              |
