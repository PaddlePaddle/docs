## [paddle 参数更多]torch.Tensor.scatter_reduce

### [torch.Tensor.scatter_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce.html#torch-tensor-scatter-reduce)

```python
torch.Tensor.scatter_reduce(dim, index, src, reduce, *, include_self=True)
```

### [paddle.Tensor.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#put-along-axis-indices-value-axis-reduce-assign-include-self-true-broadcast-true)

```python
paddle.Tensor.put_along_axis(indices, values, axis, reduce="assign", include_self=True, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| dim          | axis         | 表示 scatter 的维度，仅参数名不一致。                        |
| index        | indices      | 表示输入的索引张量，仅参数名不一致。                         |
| src          | values       | 表示需要插入的值，仅参数名不一致。                           |
| reduce       | reduce       | 表示插入 values 时的计算方式，参数默认值不一致。PyTorch 中该参数无默认值，需要输入，Paddle 中默认值为 `assign`，应设置为与 PyTorch 一致。其中 PyTorch 的 `sum` 对应 Paddle 中的 `add`，PyTorch 的 `prod` 对应 Paddle 中 `multiply`。 |
| include_self | include_self | 表示插入 values 时是否包含输入元素中的值。                   |
| -            | broadcast    | 表示是否需要广播索引张量矩阵，PyTorch 无此参数，Paddle 应设置为 `False` 才与 PyTorch 一致 |
