## [仅参数名不一致]torch.nn.ModuleList

### [torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html?highlight=torch+nn+modulelist#torch.nn.ModuleList)

```python
class torch.nn.ModuleList(modules=None)
```

### [paddle.nn.LayerList](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerList_cn.html)

```python
class paddle.nn.LayerList(sublayers=None)
```

两者功能一致，参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |             备注             |
| :-----: | :----------: | :--------------------------: |
| modules |  sublayers   | 要保存的子层，参数名不一致。 |
