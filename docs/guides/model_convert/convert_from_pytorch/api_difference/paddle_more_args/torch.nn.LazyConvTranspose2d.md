## [ paddle 参数更多 ]torch.nn.LazyConvTranspose2d
### [torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.LazyConvTranspose2d.html)

```python
torch.nn.LazyConvTranspose2d(out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
```

### [paddle.nn.Conv2DTranspose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Conv2DTranspose_cn.html#conv2dtranspose)
```python
paddle.nn.Conv2DTranspose(in_channels,
                          out_channels,
                          kernel_size,
                          stride=1,
                          padding=0,
                          output_padding=0,
                          groups=1,
                          dilation=1,
                          weight_attr=None,
                          bias_attr=None,
                          data_format='NCHW')
```

其中，Paddle 不支持 `in_channels` 参数的延迟初始化，PyTorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | in_channels   | 表示输入 Tensor 通道数，PyTorch 无此参数，Paddle 需要根据实际输入 Tensor 的通道数进行设置。   |
| out_channels  | out_channels  | 表示输出 Tensor 通道数。                           |
| kernel_size   | kernel_size   | 表示卷积核大小。                           |
| stride        | stride        | 表示卷积核步长。                           |
| padding       | padding       | 表示填充大小。                           |
| output_padding| output_padding| 表示输出 Tensor 额外添加的大小。                           |
| groups        | groups        | 表示分组数。                           |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| dilation      | dilation      | 表示空洞大小。                           |
| padding_mode  | -            | 填充模式，Paddle 无此参数，暂无转写方式。                |
| device        | -            | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。       |
| -             | weight_attr  | 指定权重参数属性的对象，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | 指定偏置参数属性的对象，当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。 |
| -             | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。              |

### 转写示例

#### in_channels: 输入通道数
在 PyTorch 中，使用 `LazyConvTranspose2d` 时可以不指定 `in_channels`，它会在第一次前向传播时根据输入 Tensor 的形状自动确定；而在 Paddle 中，创建 `Conv2DTranspose` 时必须明确指定 `in_channels` 参数，其值应与输入 Tensor 的通道数保持一致。
```python
# PyTorch 写法
conv = torch.nn.LazyConvTranspose2d(out_channels=16, kernel_size=(3, 3))
input = torch.randn(20, 5, 10, 10)  # 5 是输入通道数
output = conv(input)  # 此时 in_channels 会根据输入 Tensor 的形状自动设置为 5

# Paddle 写法
conv = paddle.nn.Conv2DTranspose(in_channels=5, out_channels=16, kernel_size=(3, 3))  # 需要明确指定 in_channels
input = paddle.randn([20, 5, 10, 10])  # 5 是输入通道数
output = conv(input)
```

#### bias: 是否在输出中添加可学习的 bias
```python
# PyTorch 写法
torch.nn.LazyConvTranspose2d(out_channels=6, kernel_size=(3, 3), bias=True)

# Paddle 写法
paddle.nn.Conv2DTranspose(in_channels=4, out_channels=6, kernel_size=(3, 3))  # in_channels 需要根据实际输入的通道数进行设置
```
```python
# PyTorch 写法
torch.nn.LazyConvTranspose2d(out_channels=6, kernel_size=(3, 3), bias=False)

# Paddle 写法
paddle.nn.Conv2DTranspose(in_channels=4, out_channels=6, kernel_size=(3, 3), bias_attr=False)  # in_channels 需要根据实际输入的通道数进行设置
```
