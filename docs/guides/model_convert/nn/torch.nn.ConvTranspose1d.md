## torch.nn.ConvTranspose1d
### [torch.nn.ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html?highlight=torch%20nn%20convtranspose1d#torch.nn.ConvTranspose1d)
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


### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| padding_mode  | -            | 填充模式。                                              |
| device        | -            | 指定 Tensor 的设备，一般对网络训练结果影响不大，可直接删除。   |
| dtype         | -            | Tensor 的所需数据类型。                                  |


### 功能差异
#### 输入格式
***PyTorch***：只支持`NCL`的输入。
***PaddlePaddle***：支持`NCL`和`NLC`两种格式的输入（通过`data_format`设置）。

#### 更新参数设置
***PyTorch***：`bias`默认为 True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。

#### padding 大小的设置
***PyTorch***：`padding`只能支持 list 或 tuple 类型。
***PaddlePaddle***：`padding`支持 list 或 tuple 类型或 str 类型。

#### padding 值的设置
***PyTorch***：通过设置`padding_mode`确定 padding 的值。
***PaddlePaddle***：PaddlePaddle 无此参数。
