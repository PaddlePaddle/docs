## [ 仅 paddle 参数更多 ]torch.Tensor.scatter_add

### [torch.Tensor.scatter_add](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add.html#torch.Tensor.scatter_add)

```python
torch.Tensor.scatter_add(dim, index, src)
```

### [paddle.Tensor.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/put_along_axis_cn.html)

```python
paddle.Tensor.put_along_axis(indices,
                              values,
                              axis,
                              reduce='assign')
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis        | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index         | indices     | 表示输入的索引张量，仅参数名不一致。                   |
| src           | values      | 表示需要插入的值，仅参数名不一致。                   |
| -             | reduce      | 表示对输出 Tensor 的计算方式， PyTorch 无此参数, Paddle 应设置为 'add' 。  |
