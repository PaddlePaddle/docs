## [ torch 参数更多 ]torch.nn.init.normal_

### [torch.nn.init.trunc_normal_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_)

```python
torch.nn.init.trunc_normal_(tensor,
                            mean=0.0,
                            std=1.0,
                            a=-2.0,
                            b=2.0,
                            generator=None)
```

### [paddle.nn.initializer.TruncatedNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/TruncatedNormal_cn.html)

```python
paddle.nn.initializer.Normal(mean=0.0,
                             std=1.0,
                             a=-2.0,
                             b=2.0,
                             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | -          | n 维 tensor。Paddle 无此参数，因为是通过调用类的 __call__ 函数来进行 tensor 的初始化。    |
| mean          |  mean          | 正态分布的平均值。参数名和默认值一致。               |
| std           |  std         | 正态分布的标准差。参数名和默认值一致。               |
| a           |  a         | 截断正态分布的下界。参数名和默认值一致。               |
| b           |  b         | 截断正态分布的上界。参数名和默认值一致。               |
| generator     |  -          | PyTorch 用于采样的生成器，默认值为 None。Paddle 无此参数，暂无转写方式。              |
