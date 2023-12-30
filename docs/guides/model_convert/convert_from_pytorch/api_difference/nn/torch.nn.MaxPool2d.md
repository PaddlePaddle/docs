## [ torch 参数更多 ]torch.nn.MaxPool2d
### [torch.nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool2d#torch.nn.MaxPool2d)

```python
torch.nn.MaxPool2d(kernel_size,
                   stride=None,
                   padding=0,
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False)
```

### [paddle.nn.MaxPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxPool2D_cn.html#maxpool2d)

```python
paddle.nn.MaxPool2D(kernel_size,
                    stride=None,
                    padding=0,
                    return_mask=False,
                    ceil_mode=False,
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size          | kernel_size            | 表示池化核大小。                           |
| stride          | stride            | 表示池化核步长。                           |
| padding          | padding            | 表示填充大小。                           |
| dilation      | -            | 设置空洞池化的大小，Paddle 无此参数，暂无转写方式。               |
| return_indices | return_mask  | 是否返回最大值的索引，仅参数名不一致。                                  |
| ceil_mode | ceil_mode  | 表示是否用 ceil 函数计算输出的 height 和 width 。                                  |
