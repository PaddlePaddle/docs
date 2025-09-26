## [ 输入参数用法不一致 ]torch.nn.Sequential

### [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential)

```python
torch.nn.Sequential(arg: OrderedDict[str, Module])
```

### [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Sequential_cn.html)

```python
paddle.nn.Sequential(*layers)
```

其中功能一致, 参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                       |
| ------- | ------------ | ------------------------------------------------------------------------------------------ |
| arg     | layers       | Paddle 支持 Layers 或可迭代的 name Layer 对，PyTorch 支持类型更多，包含 OrderedDict 类型。 |
