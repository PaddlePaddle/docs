## [ torch 参数更多 ] torch.distributions.Binomial

### [torch.distributions.Binomial](https://pytorch.org/docs/stable/distributions.html#torch.distributions.binomial.Binomial)

```python
torch.distributions.Binomial(total_count=1,
                             probs=None,
                             logits=None,
                             validate_args=None)
```

### [paddle.distribution.Binomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/distribution/Binomial_cn.html#binomial)

```python
paddle.distribution.Binomial(total_count,
                             probs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| total_count        | total_count      | 样本大小。                         |
| probs           | probs      | 每次伯努利实验中事件发生的概率。         |
| logits         | -  | 采样 1 的 log-odds，Paddle 无此参数，暂无转写方式。 |
| validate_args        | -      | 是否添加验证环节。Paddle 无此参数，一般对训练结果影响不大，可直接删除。 |
