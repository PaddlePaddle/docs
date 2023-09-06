## [torch 参数更多]torch.distributions.multinomial.Multinomial

### [torch.distributions.multinomial.Multinomial](https://pytorch.org/docs/stable/distributions.html#torch.distributions.multinomial.Multinomial)

```python
torch.distributions.multinomial.Multinomial(total_count=1, probs=None, logits=None, validate_args=None)
```

### [paddle.distribution.Multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Multinomial_cn.html)

```python
paddle.distribution.Multinomial(total_count, probs)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                           |
| ------------- | ------------ | ---------------------------------------------- |
| total_count   | total_count  | 实验次数。                                     |
| probs         | probs        | 每个类别发生的概率。                           |
| logits        | -            | 事件 log 概率，Paddle 无此参数，暂无转写方式。 |
| validate_args | -            | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。  |
