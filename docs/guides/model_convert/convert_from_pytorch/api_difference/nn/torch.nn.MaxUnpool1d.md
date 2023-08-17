## [仅 paddle 参数更多 ]torch.nn.MaxUnpool1d
### [torch.nn.MaxUnpool1d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool1d.html?highlight=maxunpool1d#torch.nn.MaxUnpool1d)

```python
torch.nn.MaxUnpool1d(kernel_size,
                     stride=None,
                     padding=0)
```

### [paddle.nn.MaxUnPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxUnPool1D_cn.html)

```python
paddle.nn.MaxUnPool1D(kernel_size,
                      stride=None,
                      padding=0,
                      data_format='NCL',
                      output_size=None,
                      name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size          | kernel_size            | 表示反池化核大小。                           |
| stride          | stride            | 表示反池化核步长。                           |
| padding          | padding            | 表示填充大小。                           |
| -             | data_format  | 输入和输出的数据格式，Pytorch 无此参数，Paddle 保持默认即可。  |
| -             | output_size  | 目标输出尺寸，Pytorch 无此参数，Paddle 保持默认即可。        |
