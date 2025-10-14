## [ paddle 参数更多 ]torch.nn.ConstantPad2d
### [torch.nn.ConstantPad2d](https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad2d.html?highlight=pad#torch.nn.ConstantPad2d)
```python
torch.nn.ConstantPad2d(padding,
                       value)
```

### [paddle.nn.Pad2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad2D_cn.html#pad2d)
```python
paddle.nn.Pad2D(padding,
                mode='constant',
                value=0.0,
                data_format='NCHW',
                name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | padding      | 填充大小|
| value             | value         | 以 'constant' 模式填充区域时填充的值。默认值为 0.0 。  |
| -             | mode         | padding 的四种模式，PyTorch 无此参数，Paddle 保持默认即可。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |
