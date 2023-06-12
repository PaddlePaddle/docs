## [torch 参数更多]torch.distributions.beta.Beta

### [torch.distributions.beta.Beta](https://pytorch.org/docs/1.13/distributions.html#torch.distributions.beta.Beta)

```python
torch.distributions.beta.Beta(concentration1, concentration0, validate_args=None)
```

### [paddle.distribution.Beta](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Beta_cn.html)

```python
paddle.distribution.Beta(alpha, beta)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                          |
| -------------- | ------------ | --------------------------------------------- |
| concentration1 | alpha        | 即公式中 α 参数，仅参数名不一致。             |
| concentration0 | beta         | 即公式中 β 参数，仅参数名不一致。             |
| validate_args  | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
