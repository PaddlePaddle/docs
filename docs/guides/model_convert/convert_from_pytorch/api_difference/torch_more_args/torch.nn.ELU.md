## [ torch 参数更多 ]torch.nn.ELU
### [torch.nn.ELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU)
```python
torch.nn.ELU(alpha=1.0,
             inplace=False)
```

### [paddle.nn.ELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ELU_cn.html#paddle.nn.ELU)
```python
paddle.nn.ELU(alpha=1.0,
              inplace=False,
              name=None)
```

两者功能一致。

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| alpha           | alpha         | 表示公式中的超参数。        |
| inplace       | inplace       | 在不更改变量的内存地址的情况下，直接修改变量的值，默认值为 False。    |
