## [torch 参数更多]torch.nn.Module.named_modules

### [torch.nn.Module.named_modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules)

```python
torch.nn.Module.named_modules(memo=None, prefix='', remove_duplicate=True)
```

### [paddle.nn.Layer.named_sublayers](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#named-sublayers-prefix-include-self-false-layers-set-none)

```python
paddle.nn.Layer.named_sublayers(prefix='', include_self=False, layers_set=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| memo          | -           | 是否用备忘录对结果进行存储。Paddle 无此参数，暂无转写方式。                               |
| prefix   | prefix  | 在所有参数名称前加的前缀。                                            |
| remove_duplicate   | -  | 是否删除结果中重复的模块实例, Paddle 无此参数，暂无转写方式。                                            |
| -         | include_self      | 是否包含该层自身，PyTorch 无此参数，Paddle 保持默认即可。                                                |
| -         | layers_set      | 记录重复子层的集合，PyTorch 无此参数，Paddle 保持默认即可。                                                |
