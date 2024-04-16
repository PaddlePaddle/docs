## [ 仅参数名不一致 ]torch.nn.Module.parameters
### [torch.nn.Module.parameters](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters)

```python
torch.nn.Module.parameters(recurse=True)
```

### [paddle.nn.Layer.parameters](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#parameters-include-sublayers-true)

```python
paddle.nn.Layer.parameters(include_sublayers=True)
```
两者功能一致，仅参数名不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| recurse       | include_sublayers    |  是否返回子层的参数，仅参数名不一致。                   |
