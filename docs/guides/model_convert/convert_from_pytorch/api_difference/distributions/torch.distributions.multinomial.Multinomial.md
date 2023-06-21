## [torch 参数更多]torch.distributions.multinomial.Multinomial

### [torch.distributions.multinomial.Multinomial](https://pytorch.org/docs/1.13/distributions.html#torch.distributions.multinomial.Multinomial)

```python
torch.distributions.multinomial.Multinomial(total_count=1, probs=None, logits=None, validate_args=None)
```

### [paddle.distribution.Multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Multinomial_cn.html)

```python
paddle.distribution.Multinomial(total_count, probs)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                           |
| ------------- | ------------ | ---------------------------------------------- |
| total_count   | total_count  | 实验次数。                                     |
| probs         | probs        | 每个类别发生的概率。                           |
| logits        | -            | 事件 log 概率，Paddle 无此参数，暂无转写方式。 |
| validate_args | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。  |
