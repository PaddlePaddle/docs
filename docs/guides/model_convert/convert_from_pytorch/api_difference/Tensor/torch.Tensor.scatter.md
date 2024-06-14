## [仅 paddle 参数更多]torch.Tensor.scatter

### [torch.Tensor.scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter.html#torch.Tensor.scatter)

```python
torch.Tensor.scatter(dim, index, src, reduce=None)
```

### [paddle.Tensor.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#put-along-axis-arr-index-value-axis-reduce-assign)

```python
paddle.Tensor.put_along_axis(indices, values, axis, reduce="assign", include_self=True, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | indices        | 表示输入的索引张量，仅参数名不一致。 |
| src     | values        | 表示需要插入的值，仅参数名不一致。 |
| reduce       | reduce       | 归约操作类型 。 |
| -            | include_self | 表示插入 values 时是否包含 arr 中的元素，PyTorch 无此参数。|
| -            | broadcast   | 表示是否需要广播 indices 矩阵，PyTorch 无此参数。 |
