## torch.nn.Hardshrink
### [torch.nn.Hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html?highlight=hardshrink#torch.nn.Hardshrink)

```python
torch.nn.Hardshrink(lambd=0.5)
```

### [paddle.nn.Hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Hardshrink_cn.html#hardshrink)

```python
paddle.nn.Hardshrink(threshold=0.5,
                        name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| lambd         | threshold    | Hardshrink 激活计算公式中的阈值。                         |
