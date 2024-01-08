## [torch 参数更多]torch.distributions.independent.Independent

### [torch.distributions.independent.Independent](https://pytorch.org/docs/stable/distributions.html#torch.distributions.independent.Independent)

```python
torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims, validate_args=None)
```

### [paddle.distribution.Independent](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Independent_cn.html)

```python
paddle.distribution.Independent(base, reinterpreted_batch_rank)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                   | PaddlePaddle             | 备注                                             |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| base_distribution         | base                     | 基础分布，仅参数名不一致。                       |
| reinterpreted_batch_ndims | reinterpreted_batch_rank | 用于转换为事件维度的批维度数量，仅参数名不一致。 |
| validate_args             | -                        | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。    |
