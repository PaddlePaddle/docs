## [torch 参数更多 ]torch.nn.Hardsigmoid
### [torch.nn.Hardsigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html?highlight=hardsigmoid#torch.nn.Hardsigmoid)

```python
torch.nn.Hardsigmoid(inplace=False)
```

### [paddle.nn.Hardsigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Hardsigmoid_cn.html#hardsigmoid)

```python
paddle.nn.Hardsigmoid(name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace       | -            | 在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
