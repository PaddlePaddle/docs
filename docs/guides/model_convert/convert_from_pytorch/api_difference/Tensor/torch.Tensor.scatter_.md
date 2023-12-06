## [仅参数名不一致]torch.Tensor.scatter_

### [torch.Tensor.scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html?highlight=scatter_#torch.Tensor.scatter_)

```python
torch.Tensor.scatter_(dim, index, src)
```

### [paddle.Tensor.put_along_axis_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/put_along_axis__cn.html#put-along-axis)

```python
paddle.Tensor.put_along_axis_(index, value, axis, reduce="assign")

```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | indices        | 表示输入的索引张量，仅参数名不一致。 |
| src     | values        | 表示需要插入的值，仅参数名不一致。 |
| reduce       | reduce       | 归约操作类型，PyTorch 默认为 None， Paddle 保持默认即可。 |
