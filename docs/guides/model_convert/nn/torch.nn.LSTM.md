## torch.nn.LSTM
### [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM)
```python
torch.nn.LSTM(input_size,
              hidden_size,
              num_layers=1,
              bias=True,
              batch_first=False,
              dropout=0,
              bidirectional=False,
              proj_size=0)
```

### [paddle.nn.LSTM](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LSTM_cn.html#lstm)
```python
paddle.nn.LSTM(input_size,
               hidden_size,
               num_layers=1,
               direction='forward',
               dropout=0.,
               time_major=False,
               weight_ih_attr=None,
               weight_hh_attr=None,
               bias_ih_attr=None,
               bias_hh_attr=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| batch_first   | time_major   | PyTorch 表示 batch size 是否为第一维，PaddlePaddle 表示 time steps 是否为第一位，它们的意义相反。  |
| bidirectional | direction    | PyTorch 表示是否进行双向 LSTM，PyTorch 使用字符串表示是双向 LSTM（`bidirectional`）还是单向 LSTM（`forward`）。 |
| proj_size   | -   | 表示 LSTM 后将映射到对应的大小，PaddlePaddle 无此功能。  |

### 功能差异

#### 映射大小的设置
***PyTorch***：支持将 LSTM 的结果映射到到对应大小，其具体方式可参见[论文](https://arxiv.org/abs/1402.1128)。
***PaddlePaddle***：无此功能。

#### 更新参数设置
***PyTorch***：`bias`默认为 True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_ih_attr`/`weight_hh_attr`/`bias_ih_attr`/`bias_hh_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/param_attr/ParamAttr_cn.html#cn-api-fluid-paramattr)；当`bias_ih_attr`/`bias_hh_attr`设置为 bool 类型与 PyTorch 的作用一致。
