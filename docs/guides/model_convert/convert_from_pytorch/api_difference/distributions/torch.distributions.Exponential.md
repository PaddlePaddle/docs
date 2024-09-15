### [ torch 参数更多 ] torch.distributions.Exponential

### [torch.distributions.Exponential](https://pytorch.org/docs/stable/distributions.html#torch.distributions.exponential.Exponential.arg_constraints)

```python
torch.distributions.Exponential(rate,
                                validate_args=None)
```

### [paddle.distribution.Exponential](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distribution/ExponentialFamily_cn.html#exponential)

```python
paddle.distribution.Exponential(rate)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | Paddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| rate           | rate      | 分布的速率参数。         |
| validate_args        | -      | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
