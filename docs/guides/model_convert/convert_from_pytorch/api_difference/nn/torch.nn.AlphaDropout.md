## [torch 参数更多]torch.nn.AlphaDropout

### [torch.nn.AlphaDropout](https://pytorch.org/docs/stable/generated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout)

```python
torch.nn.AlphaDropout(p=0.5, inplace=False)
```

### [paddle.nn.AlphaDropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AlphaDropout_cn.html)

```python
paddle.nn.AlphaDropout(p=0.5, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| p       | p            | 将输入节点置 0 的概率，即丢弃概率。                                                                             |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
