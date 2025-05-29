## [torch 参数更多 ]torch.distributions.chi2.Chi2

### [torch.distributions.chi2.Chi2](https://pytorch.org/docs/stable/distributions.html#chi2)

```python
torch.distributions.chi2.Chi2(df, validate_args=None)
```

### [paddle.distribution.Chi2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Chi2_cn.html#prob-value)

```python
paddle.distribution.Chi2(df)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                                    |
| -------------- | ------------ | ----------------------------------------------------------------------- |
| df | df        | 表示输入的参数。                                       |
| validate_args  | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
