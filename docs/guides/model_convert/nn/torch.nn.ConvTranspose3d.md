## torch.nn.ConvTranspose3d
### [torch.nn.ConvTranspose3d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html?highlight=convtranspose3d#torch.nn.ConvTranspose3d)
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
                         padding_mode='zeros')
```

### [paddle.nn.Conv3DTranspose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv3DTranspose_cn.html#conv3dtranspose)
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
                          data_format='NCDHW')
```
### 功能差异
#### 输入格式
***PyTorch***：只支持`NCDHW`的输入。
***PaddlePaddle***：支持`NCDHW`和`NDHWC`两种格式的输入（通过`data_format`设置）。

#### 更新参数设置
***PyTorch***：`bias`默认为 True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。

#### padding 大小的设置
***PyTorch***：`padding`只能支持 list 或 tuple 类型。
***PaddlePaddle***：`padding`支持 list 或 tuple 类型或 str 类型。

#### padding 值的设置
***PyTorch***：通过设置`padding_mode`确定 padding 的值。
***PaddlePaddle***：PaddlePaddle 无此参数。
