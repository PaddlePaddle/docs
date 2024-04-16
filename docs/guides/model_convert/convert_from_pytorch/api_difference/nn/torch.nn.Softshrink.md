## [ 仅参数名不一致 ]torch.nn.Softshrink
### [torch.nn.Softshrink](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html?highlight=nn+softshrink#torch.nn.Softshrink)

```python
torch.nn.Softshrink(lambd=0.5)
```

### [paddle.nn.Softshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Softshrink_cn.html#softshrink)

```python
paddle.nn.Softshrink(threshold=0.5,
                        name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| lambd         | threshold    | Softshrink 激活计算公式中的阈值，必须大于等于零，仅参数名不一致。            |
