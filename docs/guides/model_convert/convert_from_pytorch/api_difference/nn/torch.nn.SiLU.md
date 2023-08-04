## [torch 参数更多]torch.nn.SiLU

### [torch.nn.SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU)

```python
torch.nn.SiLU(inplace=False)
```

### [paddle.nn.Silu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Silu_cn.html)

```python
paddle.nn.Silu(name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
