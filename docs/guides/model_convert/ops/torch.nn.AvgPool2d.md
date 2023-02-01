## torch.nn.AvgPool2d
### [torch.nn.AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html?highlight=nn+avgpool2d#torch.nn.AvgPool2d)

```python
torch.nn.AvgPool2d(kernel_size, 
                    stride=None, 
                    padding=0, 
                    ceil_mode=False, 
                    count_include_pad=True, 
                    divisor_override=None)
```

### [paddle.nn.AvgPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AvgPool2D_cn.html#avgpool2d)

```python
paddle.nn.AvgPool2D(kernel_size, 
                    stride=None, 
                    padding=0, 
                    ceil_mode=False, 
                    exclusive=True, 
                    divisor_override=None, 
                    data_format='NCHW', 
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| count_include_pad| exclusive | 是否在平均池化模式忽略填充值。                              |
