## [ 仅参数名不一致 ]torch.Tensor.hardshrink

### [torch.Tensor.hardshrink](https://pytorch.org/docs/stable/generated/torch.Tensor.hardshrink.html?highlight=torch+tensor+hardshrink#torch.Tensor.hardshrink)

```python
torch.Tensor.hardshrink(lambd=0.5)
```

### [paddle.nn.functional.hardshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/hardshrink_cn.html#hardshrink)

```python
paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)
```

仅参数名不一致，具体如下。

### 参数映射

| PyTorch                           | PaddlePaddle                 | 备注                                                   |
|-----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> lambd </font> | <font color='red'> threshold </font> | Hardshrink 阈值，仅参数名不同。                                     |
