## [ 输入参数用法不一致 ]torch.nn.Conv3d
### [torch.nn.Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html?highlight=conv3d#torch.nn.Conv3d)

```python
torch.nn.Conv3d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None)
```

### [paddle.nn.Conv3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Conv3D_cn.html#conv3d)

```python
paddle.nn.Conv3D(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCDHW')
```


其中 PyTorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| in_channels          | in_channels            | 表示输入 Tensor 通道数。                           |
| out_channels          | out_channels            | 表示输出 Tensor 通道数。                           |
| kernel_size          | kernel_size            | 表示卷积核大小。                           |
| stride          | stride            | 表示卷积核步长。                           |
| padding          | padding            | 表示填充大小。                           |
| dilation          | dilation            | 表示空洞大小。                           |
| groups          | groups            | 表示分组数。                           |
| `bias`          | -            | 是否在输出中添加可学习的 bias。                           |
| padding_mode          | padding_mode            | 表示填充模式。                           |
| device        | -            | 指定 Tensor 的设备，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | 指定权重参数属性的对象，一般对网络训练结果影响不大，可直接删除。 |
| -             | weight_attr  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | Tensor 的所需数据类型，当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。 |
| -             | data_format  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |


### 转写示例
#### bias: 是否在输出中添加可学习的 bias
```python
# PyTorch 写法
torch.nn.Conv3D(16, 33, 3, bias=True)

# Paddle 写法
paddle.nn.Conv3D(16, 33, 3)
```
```python
# PyTorch 写法
torch.nn.Conv3D(16, 33, 3, bias=False)

# Paddle 写法
paddle.nn.Conv3D(16, 33, 3, bias_attr=False)
```
