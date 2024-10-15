## [ 仅参数名不一致 ]torch.nn.ParameterDict

### [torch.nn.ParameterDict](https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html?highlight=nn+parameterlist#torch.nn.ParameterDict)

```python
torch.nn.ParameterDict(values=None)
```

### [paddle.nn.ParameterDict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ParameterDict_cn.html#parameterdict)

```python
paddle.nn.ParameterDict(parameters=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射


| PyTorch | PaddlePaddle | 备注                                  |
| ------- | ------------ | ------------------------------------- |
| values  | parameters   | 可迭代的 Parameters，仅参数名不一致。 |
