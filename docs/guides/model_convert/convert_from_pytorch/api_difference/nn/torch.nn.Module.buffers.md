## [ 仅参数名不一致 ]torch.nn.Module.buffers

### [torch.nn.Module.buffers](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.buffers)

```python
torch.nn.Module.buffers(recurse=True)
```

### [paddle.nn.Layer.buffers](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#buffers-include-sublayers-true)

```python
paddle.nn.Layer.buffers(include_sublayers=True)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| recurse           | include_sublayers           | 表示是否返回子层的 buffers ，仅参数名不一致。               |
