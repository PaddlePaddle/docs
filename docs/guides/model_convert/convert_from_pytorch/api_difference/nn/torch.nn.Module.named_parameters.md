## [仅参数名不一致]torch.nn.Module.named_parameters

### [torch.nn.Module.named_parameters](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module+named_parameters#torch.nn.Module.named_parameters)

```python
torch.nn.Module.named_parameters(prefix='', recurse=True, remove_duplicate=True)
```

### [paddle.nn.Layer.named_parameters](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#named-parameters-prefix-include-sublayers-true)

```python
paddle.nn.Layer.named_parameters(prefix='', include_sublayers=True, remove_duplicate=True)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| prefix   | prefix  | 在所有参数名称前加的前缀。                                            |
| recurse   | include_sublayers  | 生成该模块和所有子模块的参数, 仅参数名不一致。                                            |
| remove_duplicate   | remove_duplicate  | 是否删除结果中的重复参数。|
