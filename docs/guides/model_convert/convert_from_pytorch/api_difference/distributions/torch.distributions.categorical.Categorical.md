## [torch 参数更多]torch.distributions.categorical.Categorical

### [torch.distributions.categorical.Categorical](https://pytorch.org/docs/1.13/distributions.html#torch.distributions.categorical.Categorical)

```python
torch.distributions.categorical.Categorical(probs=None, logits=None, validate_args=None)
```

### [paddle.distribution.Categorical](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Categorical_cn.html)

```python
paddle.distribution.Categorical(logits, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                          |
| ------------- | ------------ | --------------------------------------------- |
| probs         | -            | 事件概率，Paddle 无此参数，暂无转写方式。                                    |
| logits        | logits       | 类别分布对应的 logits。                       |
| validate_args | -            | 有效参数列表，Paddle 无此参数，暂无转写方式。 |
