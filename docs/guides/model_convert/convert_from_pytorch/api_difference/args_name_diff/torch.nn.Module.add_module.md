## [ 仅参数名不一致 ]torch.nn.Module.add_module

### [torch.nn.Module.add_module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module+add_module#torch.nn.Module.add_module)

```python
torch.nn.Module.add_module(name, module)
```

### [paddle.nn.Layer.add_sublayer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#add-sublayer-name-sublayer)

```python
paddle.nn.Layer.add_sublayer(name, sublayer)
```

两者功能一致，仅参数名不一致。

### 参数映射

| PyTorch | PaddlePaddle |                     备注                     |
| ----- | ----- | ------------------------------------------ |
|  name   |     name     |                 表示子层名。                 |
| module  |   sublayer   | 表示要添加到模块中的子模块，仅参数名不一致。 |
