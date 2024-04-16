## [ 参数完全一致 ] torch.nn.init.calculate_gain

### [torch.nn.init.calculate_gain](https://pytorch.org/docs/stable/nn.init.html?highlight=gain#torch.nn.init.calculate_gain)

```python
torch.nn.init.calculate_gain(nonlinearity, param=None)
```

### [paddle.nn.initializer.calculate_gain](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/calculate_gain_cn.html)

```python
paddle.nn.initializer.calculate_gain(nonlinearity, param=None)
```

两者参数和用法完全一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| nonlinearity           |  nonlinearity          | 非线性激活函数的名称，两者参数和用法完全一致。               |
| param           | param           | 某些激活函数的参数，默认为 None，两者参数和用法完全一致。               |
