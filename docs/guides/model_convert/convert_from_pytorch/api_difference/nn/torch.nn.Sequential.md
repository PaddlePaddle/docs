## [参数名不一致]torch.nn.Sequential

### [torch.nn.Sequential](https://pytorch.org/docs/1.13/generated/torch.nn.Sequential.html#torch.nn.Sequential)

```python
torch.nn.Sequential(arg: OrderedDict[str, Module])
```

### [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html)

```python
paddle.nn.Sequential(*layers)
```

其中功能一致, 仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| arg     | layers       | Layers 或可迭代的 name Layer 对，仅参数名不一致。 |
