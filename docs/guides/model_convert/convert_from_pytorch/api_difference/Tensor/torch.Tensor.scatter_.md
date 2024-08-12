## [ paddle 参数更多 ]torch.Tensor.scatter_

### [torch.Tensor.scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter.html#torch.Tensor.scatter_)

```python
torch.Tensor.scatter_(dim, index, value, *, reduce=None)
```

### [paddle.Tensor.put_along_axis_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/put_along_axis__cn.html#put-along-axis)

```python
paddle.Tensor.put_along_axis_(indices, values, axis, reduce="assign", include_self=True, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | indices        | 表示输入的索引张量，仅参数名不一致。 |
| value     | values        | 表示需要插入的值，仅参数名不一致。 |
| reduce       | reduce       | 归约操作类型 。 |
| -            | include_self | 表示插入 values 时是否包含 arr 中的元素，PyTorch 无此参数，Paddle 保持默认即可。|
| -            | broadcast   | 表示是否需要广播 indices 矩阵，PyTorch 无此参数，Paddle 应设置为 False 结果才与 pytorch 一致。 |

--------------------------------------------------------------------

### [torch.Tensor.scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter.html#torch.Tensor.scatter_)

```python
torch.Tensor.scatter_(dim, index, src, *, reduce=None)
```

### [paddle.Tensor.put_along_axis_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/put_along_axis__cn.html#put-along-axis)

```python
paddle.Tensor.put_along_axis_(indices, values, axis, reduce="assign", include_self=True, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | indices      | 表示输入的索引张量，仅参数名不一致。 |
| src     | values       | 表示需要插入的 Tensor 值，仅参数名不一致。 |
| reduce  | reduce       | 归约操作类型 。 |
| -       | include_self | 表示插入 values 时是否包含 arr 中的元素，PyTorch 无此参数，Paddle 保持默认即可。|
| -       | broadcast    | 表示是否需要广播 indices 矩阵，PyTorch 无此参数，Paddle 应设置为 False 结果才与 pytorch 一致。 |
