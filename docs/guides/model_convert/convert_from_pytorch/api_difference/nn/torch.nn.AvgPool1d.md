# [ 参数不一致 ]torch.nn.AvgPool1d
### [torch.nn.AvgPool1d](https://pytorch.org/docs/1.13/generated/torch.nn.AvgPool1d.html?highlight=avgpool1d#torch.nn.AvgPool1d)

```python
torch.nn.AvgPool1d(kernel_size,
                   stride=None,
                   padding=0,
                   ceil_mode=False,
                   count_include_pad=True)
```

### [paddle.nn.AvgPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AvgPool1D_cn.html#avgpool1d)

```python
paddle.nn.AvgPool1D(kernel_size,
                    stride=None,
                    padding=0,
                    exclusive=True,
                    ceil_mode=False,
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
| count_include_pad | -         | 是否使用额外 padding 的值计算平均池化结果，默认为 True 。  |
| -             | exclusive    | 是否不使用额外 padding 的值计算平均池化结果，默认为 True 。        |

### 转写示例
#### count_include_pad：是否使用额外 padding 的值计算平均池化结果
```python
# Pytorch 写法
torch.nn.AvgPool1D(kernel_size=2, stride=2, count_include_pad=True)

# Paddle 写法
paddle.nn.AvgPool1D(kernel_size=2, stride=2, exclusive=False)
```
