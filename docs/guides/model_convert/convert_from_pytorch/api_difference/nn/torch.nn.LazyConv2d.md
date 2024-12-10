## [ paddle 参数更多 ]torch.nn.LazyConv2d
### [torch.nn.LazyConv2d](https://pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html)

```python
torch.nn.LazyConv2d(out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

### [paddle.nn.Conv2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Conv2D_cn.html#conv2d)

```python
paddle.nn.Conv2D(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW')
```

其中，Paddle 不支持 `in_channels` 参数的延迟初始化，PyTorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -             | in_channels            | 表示输入 Tensor 通道数，PyTorch 无此参数，Paddle 需要根据实际输入 Tensor 的通道数进行设置。                           |
| out_channels  | out_channels            | 表示输出 Tensor 通道数。                           |
| kernel_size   | kernel_size            | 表示卷积核大小。                           |
| stride        | stride            | 表示卷积核步长。                           |
| padding       | padding            | 表示填充大小。                           |
| dilation      | dilation            | 表示空洞大小。                           |
| groups        | groups            | 表示分组数。                           |
| bias          | -            | 是否在输出中添加可学习的 bias。                           |
| padding_mode          | padding_mode            | 表示填充模式。                           |
| device        | -            | 指定 Tensor 的设备，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | 指定权重参数属性的对象，一般对网络训练结果影响不大，可直接删除。 |
| -             | weight_attr  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | Tensor 的所需数据类型，当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。 |
| -             | data_format  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |


### 转写示例

#### in_channels: 输入通道数
在 PyTorch 中，使用 `LazyConv2d` 时可以不指定 `in_channels`，它会在第一次前向传播时根据输入 Tensor 的形状自动确定；而在 Paddle 中，创建 `Conv2D` 时必须明确指定 `in_channels` 参数，其值应与输入 Tensor 的通道数保持一致。
```python
# PyTorch 写法
conv = torch.nn.LazyConv2d(out_channels=16, kernel_size=3)
input = torch.randn(20, 5, 10, 10)  # 5 是输入通道数
output = conv(input)  # 此时 in_channels 会根据输入 Tensor 的形状自动设置为 5

# Paddle 写法
conv = paddle.nn.Conv2D(in_channels=5, out_channels=16, kernel_size=3)  # 需要明确指定 in_channels
input = paddle.randn([20, 5, 10, 10])  # 5 是输入通道数
output = conv(input)
```

#### bias: 是否在输出中添加可学习的 bias
```python
# PyTorch 写法
torch.nn.LazyConv2d(out_channels=33, kernel_size=3, bias=True)

# Paddle 写法
paddle.nn.Conv2D(in_channels=16, out_channels=33, kernel_size=3)  # in_channels 需要根据实际输入的通道数进行设置
```
```python
# PyTorch 写法
torch.nn.LazyConv2d(out_channels=33, kernel_size=3, bias=False)

# Paddle 写法
paddle.nn.Conv2D(in_channels=16, out_channels=33, kernel_size=3, bias_attr=False)  # in_channels 需要根据实际输入的通道数进行设置
```
