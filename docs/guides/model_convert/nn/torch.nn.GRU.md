## torch.nn.GRU
### [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=torch%20nn%20gru#torch.nn.GRU)
```python
torch.nn.GRU(input_size,
             hidden_size,
             num_layers=1,
             bias=True,
             batch_first=False,
             dropout=0,
             bidirectional=False)
```

### [paddle.nn.GRU](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GRU_cn.html#gru)
```python
paddle.nn.GRU(input_size,
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

### 功能差异
#### 更新参数设置
***PyTorch***：`bias`默认为 True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_ih_attr`/`weight_hh_attr`/`bias_ih_attr`/`bias_hh_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_ih_attr`/`bias_hh_attr`设置为 bool 类型与 PyTorch 的作用一致。
