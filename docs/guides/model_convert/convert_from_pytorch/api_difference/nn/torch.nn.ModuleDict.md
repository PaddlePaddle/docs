## [ 仅参数名不一致 ]torch.nn.ModuleDict
### [torch.nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html?highlight=nn+moduledict#torch.nn.ModuleDict)

```python
class torch.nn.ModuleDict(modules=None)
```

### [paddle.nn.LayerDict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerDict_cn.html#layerdict)

```python
class paddle.nn.LayerDict(sublayers=None)
```
两者功能一致，参数名不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| modules       | sublayers    | 键值对的可迭代对象，值的类型为 paddle.nn.Layer ，参数名不一致。                   |
