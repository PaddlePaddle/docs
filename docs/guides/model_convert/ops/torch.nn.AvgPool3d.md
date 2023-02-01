## torch.nn.AvgPool3d
### [torch.nn.AvgPool3d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html?highlight=avgpool3d#torch.nn.AvgPool3d)

```python
torch.nn.AvgPool3d(kernel_size, 
                    stride=None, 
                    padding=0, 
                    ceil_mode=False, 
                    count_include_pad=True, 
                    divisor_override=None)
```

### [paddle.nn.AvgPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AvgPool3D_cn.html#avgpool3d)

```python
paddle.nn.AvgPool3D(kernel_size, 
                    stride=None, 
                    padding=0, 
                    ceil_mode=False, 
                    exclusive=True, 
                    divisor_override=None, 
                    data_format='NCDHW', 
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| count_include_pad| exclusive | 是否在平均池化模式忽略填充值。                              |
