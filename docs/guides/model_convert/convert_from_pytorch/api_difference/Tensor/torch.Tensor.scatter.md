## [仅 paddle 参数更多]torch.Tensor.scatter

### [torch.Tensor.scatter](https://pytorch.org/docs/1.13/generated/torch.Tensor.scatter.html#torch.Tensor.scatter)

```python
torch.Tensor.scatter(dim, index, src)
```

### [paddle.Tensor.put_along_axis](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#put-along-axis-arr-index-value-axis-reduce-assign)

```python
paddle.Tensor.put_along_axis(index, value, axis, reduce="assign")

```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注    |
| ------- | ------------ | ------- |
| dim     | axis         | 表示在哪一个维度 scatter ，仅参数名不一致。 |
| index   | index        | 表示输入的索引张量，仅参数名不一致。 |
| src     | value        | 表示需要插入的值，仅参数名不一致。 |
| -       | reduce       | 归约操作类型，PyTorch 无此参数， Paddle 保持默认即可。 |
