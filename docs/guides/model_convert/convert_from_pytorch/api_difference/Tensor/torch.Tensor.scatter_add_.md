## [ 仅 paddle 参数更多 ]torch.Tensor.scatter_add_

### [torch.Tensor.scatter_add_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_)

```python
torch.Tensor.scatter_add_(dim,
                         index,
                         src)
```

### [paddle.Tensor.put_along_axis_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/put_along_axis__cn.html)

```python
paddle.Tensor.put_along_axis_(indices,
                              values,
                              axis,
                              reduce='assign',
                              include_self=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis        | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index         | indices     | 表示输入的索引张量，仅参数名不一致。                   |
| src           | values      | 表示需要插入的值，仅参数名不一致。                   |
| -             | reduce      | 表示对输出 Tensor 的计算方式， PyTorch 无此参数, Paddle 应设置为 'add' 。  |
| -            | include_self | 表示插入 values 时是否包含 arr 中的元素，PyTorch 无此参数，Paddle 应设置为 'True' 。 |
