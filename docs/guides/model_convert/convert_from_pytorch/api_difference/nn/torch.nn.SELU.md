## [torch 参数更多]torch.nn.SELU

### [torch.nn.SELU](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html#torch.nn.SELU)

```python
torch.nn.SELU(inplace=False)
```

### [paddle.nn.SELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SELU_cn.html)

```python
paddle.nn.SELU(scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -       | scale        | selu 激活计算公式中的 scale 值，PyTorch 无此参数，Paddle 保持默认即可。                                         |
| -       | alpha        | selu 激活计算公式中的 alpha 值，PyTorch 无此参数，Paddle 保持默认即可。                                         |
