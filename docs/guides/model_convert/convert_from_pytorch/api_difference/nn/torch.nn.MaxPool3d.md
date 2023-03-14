## torch.nn.MaxPool3d
### [torch.nn.MaxPool3d](https://pytorch.org/docs/1.13/generated/torch.nn.MaxPool3d.html?highlight=maxpool3d#torch.nn.MaxPool3d)

```python
torch.nn.MaxPool3d(kernel_size,
                   stride=None,
                   padding=0,
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False)
```

### [paddle.nn.MaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MaxPool3D_cn.html#maxpool3d)

```python
paddle.nn.MaxPool3D(kernel_size,
                    stride=None,
                    padding=0,
                    ceil_mode=False,
                    return_mask=False,
                    data_format='NCDHW',
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dilation      | -            | 设置空洞池化的大小，PaddlePaddle 无此参数，无转写方式。      |
| return_indices| return_mask  | 是否返回最大值的索引。                                  |
| -             | data_format  | 输入和输出的数据格式，Pytorch 无此参数，Paddle 保持默认即可。  |
