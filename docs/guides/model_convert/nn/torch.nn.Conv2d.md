# torch.nn.Conv2d
### [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d)

```python
torch.nn.Conv2d(in_channels,
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

### [paddle.nn.Conv2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html#conv2d)

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


### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| device        | -            | 指定 Tensor 的设备，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型。                                  |


### 功能差异

#### 输入格式
***PyTorch***：只支持`NCHW`的输入。
***PaddlePaddle***：支持`NCHW`和`NHWC`两种格式的输入（通过`data_format`设置）。

#### 更新参数设置
***PyTorch***：`bias`默认为 True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。

#### padding 的设置
***PyTorch***：`padding`只能支持 list 或 tuple 类型。它可以有 3 种格式：
(1)包含 4 个二元组：\[\[0,0\], \[0,0\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，其中每个元组都可使用整数值替换，代表元组中的 2 个值相等；
(2)包含 2 个二元组：\[\[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，其中每个元组都可使用整数值替换，代表元组中的 2 个值相等；
(3)包含一个整数值，padding_height = padding_width = padding。
***PaddlePaddle***：`padding`支持 list 或 tuple 类型或 str 类型。如果它是一个 list 或 tuple，它可以有 4 种格式：
(1)包含 4 个二元组：当 data_format 为"NCHW"时为 \[\[0,0\], \[0,0\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，当 data_format 为"NHWC"时为\[\[0,0\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\], \[0,0\]\]；
(2)包含 4 个整数值：\[padding_height_top, padding_height_bottom, padding_width_left, padding_width_right\]；
(3)包含 2 个整数值：\[padding_height, padding_width\]，此时 padding_height_top = padding_height_bottom = padding_height， padding_width_left = padding_width_right = padding_width；
(4)包含一个整数值，padding_height = padding_width = padding。如果它为一个字符串时，可以是"VALID"或者"SAME"，表示填充算法。
