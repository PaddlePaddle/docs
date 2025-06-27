## [ paddle 参数更多 ]torch.nn.LPPool1d

### [torch.nn.LPPool1d](https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#lppool1d)

```python
torch.nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)
```

### [paddle.nn.LPPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LPPool1D_cn.html#lppool1d)
```python
paddle.nn.LPPool1D(norm_type, kernel_size, stride=None, padding=0, ceil_mode=False, data_format='NCL', name=None)
```

其中 Paddle 参数更多，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| norm_type     | norm_type    | 幂平均池化的指数，不可以为 0 。 |
| kernel_size   | kernel_size  | 池化核的尺寸大小。|
| stride        | stride       | 池化操作步长。|
| ceil_mode     | ceil_mode    | 是否用 `ceil` 函数计算输出的 height 和 width，如果设置为 `False`，则使用 `floor` 函数来计算，默认为 `False`。|
| -             | padding      | 池化补零的方式。PyTorch 无此参数，Paddle 保持默认即可。|
| -             | data_format  | 输入和输出的数据格式，可以是"NCL"和"NLC"。PyTorch 无此参数，Paddle 保持默认即可。|
