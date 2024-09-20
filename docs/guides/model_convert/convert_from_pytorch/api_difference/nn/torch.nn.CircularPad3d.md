## [ paddle 参数更多 ]torch.nn.CircularPad3d

### [torch.nn.CircularPad3d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad3d.html#circularpad3d)

```python
torch.nn.CircularPad3d(padding)
```

### [paddle.nn.Pad3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad3D_cn.html#pad3d)

```python
paddle.nn.Pad3D(padding, mode='constant', value=0.0, data_format='NCDHW', name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                         |
| ------------- | ------------ | ---------------------------------------------------------------------------- |
| padding       | padding      | 填充大小。                                                                   |
| -             | mode         | `padding` 的四种模式。PyTorch 无此参数，Paddle 应设置为 `constant`。         |
| -             | value        | 以 `constant` 模式填充区域时填充的值。PyTorch 无此参数，Paddle 保持默认即可。|
| -             | data_format  | 输入的数据格式。PyTorch 无此参数，Paddle 保持默认即可。                      |
