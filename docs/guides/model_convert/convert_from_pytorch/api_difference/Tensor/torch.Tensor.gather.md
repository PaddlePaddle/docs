## [ paddle 参数更多 ]torch.Tensor.gather

### [torch.Tensor.gather](https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html?highlight=gather#torch.Tensor.gather)

```python
torch.Tensor.gather(dim, index)
```

### [paddle.Tensor.take_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#take-along-axis-arr-indices-axis-broadcast-true)

```python
paddle.Tensor.take_along_axis(indices, axis, broadcast=True)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                      |
| ------- | ------------ | ----------------------------------------- |
| dim     | axis         | 指定 index 获取输入的维度，仅参数名不一致。 |
| index   | indices      | 索引 Tensor，仅参数名不一致。              |
| -       | broadcast    | 表示是否需要广播 indices 矩阵，PyTorch 无此参数，Paddle 应设置为 False 结果才与 pytorch 一致。 |
