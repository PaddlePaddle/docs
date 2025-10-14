## [ paddle 参数更多 ]torch.nn.ConstantPad1d
### [torch.nn.ConstantPad1d](https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html?highlight=constantpad1d#torch.nn.ConstantPad1d)
```python
torch.nn.ConstantPad1d(padding,
                       value)
```

### [paddle.nn.Pad1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad1D_cn.html#pad1d)
```python
paddle.nn.Pad1D(padding,
                mode='constant',
                value=0.0,
                data_format='NCL',
                name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小 |
| value             | value         | 以 'constant' 模式填充区域时填充的值。默认值为 0.0 。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |
