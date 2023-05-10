## [ 参数不一致 ]torch.nn.AvgPool3d
### [torch.nn.AvgPool3d](https://pytorch.org/docs/1.13/generated/torch.nn.AvgPool3d.html?highlight=avgpool3d#torch.nn.AvgPool3d)

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

其中 Pytorch 的 count_include_pad 与 Paddle 的 exclusive 用法不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size          | kernel_size         | 表示池化核的尺寸大小 。                                     |
| stride          | stride         | 表示步长 。                                     |
| padding          | padding         | 表示填充大小 。                                     |
| ceil_mode          | ceil_mode         | 表示是否用 ceil 函数计算输出的 height 和 width 。                                     |
| count_include_pad | -         | 是否使用额外 padding 的值计算平均池化结果，默认为 True 。 Paddle 无此参数，需要转写。  |
| divisor_override | divisor_override  | 如果指定，它将用作除数，否则根据 kernel_size 计算除数。默认 None 。 |
| -             | exclusive    | 是否不使用额外 padding 的值计算平均池化结果，默认为 True。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |

### 转写示例
#### count_include_pad：是否使用额外 padding 的值计算平均池化结果
```python
# Pytorch 写法
torch.nn.AvgPool3D(kernel_size=2, stride=2, count_include_pad=True)

# Paddle 写法
paddle.nn.AvgPool3D(kernel_size=2, stride=2, exclusive=False)
```
