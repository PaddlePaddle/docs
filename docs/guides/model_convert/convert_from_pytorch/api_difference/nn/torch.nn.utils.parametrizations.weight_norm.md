
## [ 仅参数名不一致 ]torch.nn.utils.parametrizations.weight_norm

### [torch.nn.utils.parametrizations.weight_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html)

```python
torch.nn.utils.parametrizations.weight_norm(module, name='weight', dim=0)
```

### [paddle.nn.utils.weight_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/weight_norm_cn.html#weight-norm)

```python
paddle.nn.utils.weight_norm(layer, name='weight', dim=0)
```

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module  | layer | 要添加权重归一化的层。|
| name | name| 权重参数的名字。|
| dim     | dim| 进行归一化操作的切片所在维度。|
