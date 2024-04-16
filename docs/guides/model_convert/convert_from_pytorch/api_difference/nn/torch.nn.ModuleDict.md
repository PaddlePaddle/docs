## [ 仅参数名不一致 ]torch.nn.ModuleDict

### [torch.nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html?highlight=torch+nn+moduledict#torch.nn.ModuleDict)

```python
torch.nn.ModuleDict(modules=None)
```

### [paddle.nn.LayerDict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LayerDict_cn.html)

```python
paddle.nn.LayerDict(sublayers=None)
```

两者功能一致，参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
|  -----  |  ----------  |  ---------------------------------------------------------- |
| modules |  sublayers   | 键值对的可迭代对象，值的类型为 paddle.nn.Layer ，参数名不一致。 |
