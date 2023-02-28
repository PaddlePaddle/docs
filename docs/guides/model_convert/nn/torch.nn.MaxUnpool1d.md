## torch.nn.MaxUnpool1d
### [torch.nn.MaxUnpool1d](https://pytorch.org/docs/1.13/generated/torch.nn.MaxUnpool1d.html?highlight=maxunpool1d#torch.nn.MaxUnpool1d)

```python
torch.nn.MaxUnpool1d(kernel_size,
                     stride=None,
                     padding=0)
```

### [paddle.nn.MaxUnpool1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MaxUnPool1D_cn.html#maxunpool1d)

```python
paddle.nn.MaxUnpool1d(kernel_size,
                      stride=None,
                      padding=0,
                      data_format='NCL',
                      output_size=None,
                      name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | data_format  | 输入和输出的数据格式，Pytorch 无此参数，Paddle 保持默认即可。  |
| -             | output_size  | 目标输出尺寸，Pytorch 无此参数，Paddle 保持默认即可。        |
