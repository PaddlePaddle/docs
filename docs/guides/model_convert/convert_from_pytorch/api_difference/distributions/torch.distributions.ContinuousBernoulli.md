## [ torch 参数更多 ] torch.distributions.ContinuousBernoulli

### [torch.distributions.ContinuousBernoulli](https://pytorch.org/docs/stable/distributions.html)

```python
torch.distributions.ContinuousBernoulli(probs=None,
                                        logits=None,
                                        lims=(0.499, 0.501)
                                        validate_args=None)
```

### [paddle.distribution.ContinuousBernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distribution/ContinuousBernoulli_cn.html#continuousbernoulli)

```python
paddle.distribution.ContinuousBernoulli(probs,
                                        lims=(0.499, 0.501))
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| probs           | probs      | 参数化分布的 (0,1) 值。         |
| logits         | -  | 实值参数，与 probs 通过 sigmoid 函数匹配。Paddle 无此参数，暂无转写方式。 |
| lims       | lims      | 一个包含两个元素的元组，指定了分布的下限和上限，默认为 (0.499, 0.501)。                         |
| validate_args        | -      | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
