## [ 仅参数名不一致 ]torch.nn.Hardshrink
### [torch.nn.Hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html?highlight=hardshrink#torch.nn.Hardshrink)

```python
torch.nn.Hardshrink(lambd=0.5)
```

### [paddle.nn.Hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardshrink_cn.html#hardshrink)

```python
paddle.nn.Hardshrink(threshold=0.5,
                        name=None)
```
两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| lambd         | threshold    | Hardshrink 激活计算公式中的阈值，仅参数名不一致。                         |
