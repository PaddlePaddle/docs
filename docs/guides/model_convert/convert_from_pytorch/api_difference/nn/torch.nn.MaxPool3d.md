## [ torch 参数更多 ]torch.nn.MaxPool3d
### [torch.nn.MaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html?highlight=maxpool3d#torch.nn.MaxPool3d)

```python
torch.nn.MaxPool3d(kernel_size,
                   stride=None,
                   padding=0,
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False)
```

### [paddle.nn.MaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxPool3D_cn.html#maxpool3d)

```python
paddle.nn.MaxPool3D(kernel_size,
                    stride=None,
                    padding=0,
                    ceil_mode=False,
                    return_mask=False,
                    data_format='NCDHW',
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size          | kernel_size            | 表示池化核大小。                           |
| stride          | stride            | 表示池化核步长。                           |
| padding          | padding            | 表示填充大小。                           |
| dilation      | -            | 设置空洞池化的大小，Paddle 无此参数，暂无转写方式。               |
| return_indices | return_mask  | 是否返回最大值的索引，仅参数名不一致。                                  |
| ceil_mode | ceil_mode  | 表示是否用 ceil 函数计算输出的 height 和 width 。                                  |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。  |
