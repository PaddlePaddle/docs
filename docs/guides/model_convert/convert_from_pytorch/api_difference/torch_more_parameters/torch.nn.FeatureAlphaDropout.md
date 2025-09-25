## [torch 参数更多]torch.nn.FeatureAlphaDropout

### [torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/stable/generated/torch.nn.FeatureAlphaDropout.html#torch.nn.FeatureAlphaDropout)

```python
torch.nn.FeatureAlphaDropout(p=0.5, inplace=False)
```

### [paddle.nn.FeatureAlphaDropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/FeatureAlphaDropout_cn.html)

```python
paddle.nn.FeatureAlphaDropout(p=0.5, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| p       | p            | 将输入节点置 0 的概率，即丢弃概率。                                                                             |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
