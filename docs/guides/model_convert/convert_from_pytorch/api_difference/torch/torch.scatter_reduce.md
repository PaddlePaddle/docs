## [paddle 参数更多]torch.scatter_reduce

### [torch.scatter_reduce](https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html#torch-scatter-reduce)

```python
torch.scatter_reduce(input, dim, index, src, reduce, *, include_self=True)
```

### [paddle.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/put_along_axis_cn.html)

```python
paddle.put_along_axis(arr, indices, values, axis, reduce='assign', include_self=True, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| input        | arr          | 表示输入 Tensor，仅参数名不一致。                            |
| dim          | axis         | 表示沿着哪个维度 scatter，仅参数名不一致。                   |
| index        | indices      | 表示输入的索引张量，仅参数名不一致。                         |
| src          | values       | 表示要插入的值，仅参数名不一致。                             |
| reduce       | reduce       | 表示插入 values 时的计算方式，参数默认值不一致。PyTorch 中该参数无默认值，需要输入，Paddle 中默认值为 `assign`，应设置为与 PyTorch 一致。其中 PyTorch 的 `sum` 对应 Paddle 中的 `add`，PyTorch 的 `prod` 对应 Paddle 中 `multiple`。 |
| include_self | include_self | 表示插入 values 时是否包含输入 Tensor 中的元素。             |
| -            | broadcast    | 表示是否需要广播输入的索引张量，PyTorch 无此参数，Paddle 应设置为 `False` 结果才与 pytorch 一致。 |
