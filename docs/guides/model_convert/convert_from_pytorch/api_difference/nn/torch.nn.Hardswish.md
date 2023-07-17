## [torch 参数更多]torch.nn.Hardswish

### [torch.nn.Hardswish](https://pytorch.org/docs/1.13/generated/torch.nn.Hardswish.html#torch.nn.Hardswish)

```python
torch.nn.Hardswish(inplace=False)
```

### [paddle.nn.Hardswish](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardswish_cn.html)

```python
paddle.nn.Hardswish(name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                        |
| ------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| inplace | -            | 在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
