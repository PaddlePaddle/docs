## [ 仅参数名不一致 ]torch.nn.Module.named_buffers

### [torch.nn.Module.named_buffers](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)

```python
torch.nn.Module.named_buffers(prefix='', recurse=True, remove_duplicate=True)
```

### [paddle.nn.Layer.named_buffers](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#named-buffers-prefix-include-sublayers-true)

```python
paddle.nn.Layer.named_buffers(prefix='', include_sublayers=True, remove_duplicate=True)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| prefix   | prefix  | 在所有参数名称前加的前缀。                                            |
| recurse          | include_self           | 生成该模块和所有子模块的缓冲区，仅参数名不一致。                               |
| remove_duplicate   | remove_duplicate  | 是否删除结果中重复的模块实例。                                        |
