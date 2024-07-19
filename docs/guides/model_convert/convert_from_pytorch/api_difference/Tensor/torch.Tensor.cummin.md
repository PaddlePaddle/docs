## [ paddle 参数更多 ]torch.Tensor.cummin

### [torch.Tensor.cummin](https://pytorch.org/docs/stable/generated/torch.Tensor.cummin.html?highlight=cummin#torch.Tensor.cummin)

```python
torch.Tensor.cummin(dim)
```

### [paddle.Tensor.cummin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cummin-x-axis-none-dtype-int64-name-none)

```python
paddle.Tensor.cummin(axis=None, dtype=None, name=None)
```

两者功能一致，其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                           |
| ------- | ------------ | ------------------------------ |
| dim     | axis         | 需要累加的维度，仅参数名不一致。 |
| -   | dtype        | 输出 Tensor 的数据类型。PyTorch 无此参数， Paddle 保持默认即可。       |
