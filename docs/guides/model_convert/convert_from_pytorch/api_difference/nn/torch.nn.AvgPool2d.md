## torch.nn.AvgPool2d
### [torch.nn.AvgPool2d](https://pytorch.org/docs/1.13/generated/torch.nn.AvgPool2d.html?highlight=avgpool2d#torch.nn.AvgPool2d)

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

其中 Pytorch 的 count_include_pad 与 Paddle 的 exclusive 用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| count_include_pad| -         | 是否使用额外 padding 的值计算平均池化结果，默认为 True。  |
| -             | exclusive    | 是否不使用额外 padding 的值计算平均池化结果，默认为 True。  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |

### 转写示例
#### count_include_pad：是否使用额外 padding 的值计算平均池化结果
```python
# Pytorch 写法
torch.nn.AvgPool2D(kernel_size=2, stride=2, count_include_pad=True)

# Paddle 写法
paddle.nn.AvgPool2D(kernel_size=2, stride=2, exclusive=False)
```
