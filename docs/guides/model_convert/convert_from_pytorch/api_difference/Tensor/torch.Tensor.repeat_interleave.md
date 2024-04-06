## [ torch 参数更多]torch.Tensor.repeat_interleave

### [torch.Tensor.repeat_interleave](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat_interleave.html#torch.Tensor.repeat_interleave)

```python
torch.Tensor.repeat_interleave(repeats, dim=None, *, output_size=None)
```

### [paddle.Tensor.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#repeat-interleave-repeats-axis-none-name-none)

```python
paddle.Tensor.repeat_interleave(repeats, axis=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| repeats   | repeats    | 表示指定复制次数的 1-D Tensor 或指定的复制次数。           |
| dim     |   axis        | 表示复制取值的维度，仅参数名不一致。 |
| output_size     | -        | 表示给定维度的总输出尺寸，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
