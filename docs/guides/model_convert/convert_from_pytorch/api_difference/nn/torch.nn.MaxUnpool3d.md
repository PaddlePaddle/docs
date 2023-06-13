## [仅 paddle 参数更多 ]torch.nn.MaxUnpool3d
### [torch.nn.MaxUnpool3d](https://pytorch.org/docs/1.13/generated/torch.nn.MaxUnpool3d.html?highlight=maxunpool3d#torch.nn.MaxUnpool3d)

```python
torch.nn.MaxUnpool3d(kernel_size,
                     stride=None,
                     padding=0)
```

### [paddle.nn.MaxUnPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MaxUnPool3D_cn.html)

```python
paddle.nn.MaxUnPool3D(kernel_size,
                      stride=None,
                      padding=0,
                      data_format='NCDHW',
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
