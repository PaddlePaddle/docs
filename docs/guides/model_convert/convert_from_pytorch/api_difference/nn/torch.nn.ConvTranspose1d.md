## torch.nn.ConvTranspose1d
### [torch.nn.ConvTranspose1d](https://pytorch.org/docs/1.13/generated/torch.nn.ConvTranspose1d.html?highlight=convtranspose1d#torch.nn.ConvTranspose1d)
```python
torch.nn.ConvTranspose1d(in_channels,
                         out_channels,
                         kernel_size,
                         stride=1,
                         padding=0,
                         output_padding=0,
                         groups=1,
                         bias=True,
                         dilation=1,
                         padding_mode='zeros',
                         device=None,
                         dtype=None)
```

### [paddle.nn.Conv1DTranspose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv1DTranspose_cn.html#conv1dtranspose)
```python
paddle.nn.Conv1DTranspose(in_channels,
                          out_channels,
                          kernel_size,
                          stride=1,
                          padding=0,
                          output_padding=0,
                          groups=1,
                          dilation=1,
                          weight_attr=None,
                          bias_attr=None,
                          data_format='NCL')
```

其中 Pytorch 的 bias 与 Paddle 的 bias_attr 用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| padding_mode  | -            | 填充模式，PaddlePaddle 无此参数，无转写方式。                                              |
| device        | -            | 指定 Tensor 的设备，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型，PaddlePaddle 无此参数，无转写方式。                                  |
| -             | weight_attr  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | 指定偏置参数属性的对象，当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。 |
| -             | data_format  | 输入和输出的数据格式，Pytorch 无此参数，Paddle 保持默认即可。                                  |


### 转写示例
#### bias: 是否在输出中添加可学习的 bias
```python
# Pytorch 写法
torch.nn.ConvTranspose1d(2, 1, 2, bias=True)

# Paddle 写法
paddle.nn.Conv1DTranspose(2, 1, 2)
```
```python
# Pytorch 写法
torch.nn.ConvTranspose1d(2, 1, 2, bias=False)

# Paddle 写法
paddle.nn.Conv1DTranspose(2, 1, 2, bias_attr=False)
```
