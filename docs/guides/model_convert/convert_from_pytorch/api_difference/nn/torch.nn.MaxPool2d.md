## torch.nn.MaxPool2d
### [torch.nn.MaxPool2d](https://pytorch.org/docs/1.13/generated/torch.nn.MaxPool2d.html?highlight=maxpool2d#torch.nn.MaxPool2d)

```python
torch.nn.MaxPool2d(kernel_size,
                   stride=None,
                   padding=0,
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False)
```

### [paddle.nn.MaxPool2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MaxPool2D_cn.html#maxpool2d)

```python
paddle.nn.MaxPool2D(kernel_size,
                    stride=None,
                    padding=0,
                    return_mask=False,
                    ceil_mode=False,
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dilation      | -            | 设置空洞池化的大小，PaddlePaddle 无此参数，无转写方式。               |
| return_indices| return_mask  | 是否返回最大值的索引。                                  |
