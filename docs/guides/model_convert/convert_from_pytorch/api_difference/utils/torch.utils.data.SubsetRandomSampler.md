## [ torch 参数更多 ] torch.utils.data.SubsetRandomSampler

### [torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler)

```python
torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

### [paddle.io.SubsetRandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/SubsetRandomSampler_cn.html#paddle.io.SubsetRandomSampler)

```python
paddle.io.SubsetRandomSampler(indices)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                 |
| ----------- | ------------ | -------------------------------------------------------------------- |
| indices     | indices      | 子集在原数据集中的索引序列，需要是 list 或者 tuple 类型。            |
| generator   | -            | Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。            |
