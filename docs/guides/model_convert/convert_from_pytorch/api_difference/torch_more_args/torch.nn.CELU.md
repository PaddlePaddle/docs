## [torch 参数更多]torch.nn.CELU

### [torch.nn.CELU](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html#torch.nn.CELU)

```python
torch.nn.CELU(alpha=1.0, inplace=False)
```

### [paddle.nn.CELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/CELU_cn.html)

```python
paddle.nn.CELU(alpha=1.0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| alpha   | alpha        | CELU 的 alpha 值。                                                                                              |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
