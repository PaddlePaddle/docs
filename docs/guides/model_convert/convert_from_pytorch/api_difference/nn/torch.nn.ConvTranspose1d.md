## [ 参数不一致 ]torch.nn.ConvTranspose1d
### [torch.nn.ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html?highlight=convtranspose1d#torch.nn.ConvTranspose1d)
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

其中 Pytorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| in_channels          | in_channels            | 表示输入 Tensor 通道数。                           |
| out_channels          | out_channels            | 表示输出 Tensor 通道数。                           |
| kernel_size          | kernel_size            | 表示卷积核大小。                           |
| stride          | stride            | 表示卷积核步长。                           |
| padding          | padding            | 表示填充大小。                           |
| output_padding          | output_padding            | 表示输出 Tensor 额外添加的大小。                           |
| groups          | groups            | 表示分组数。                           |
| `bias`          | -            | 是否在输出中添加可学习的 bias。                             |
| dilation          | dilation            | 表示空洞大小。                           |
| padding_mode  | -            | 填充模式，Paddle 无此参数，暂无转写方式。                |
| device        | -            | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。       |
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
