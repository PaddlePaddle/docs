## [ 输入参数用法不一致 ]torch.nn.GRU
### [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=torch%20nn%20gru#torch.nn.GRU)
```python
torch.nn.GRU(input_size,
             hidden_size,
             num_layers=1,
             bias=True,
             batch_first=False,
             dropout=0,
             bidirectional=False,
             device=None,
             dtype=None)
```

### [paddle.nn.GRU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GRU_cn.html#gru)
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

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| num_layers          | num_layers            | 表示循环网络的层数。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持，需要转写。  |
| batch_first   | time_major   | PyTorch 表示 batch size 是否为第一维，PaddlePaddle 表示 time steps 是否为第一维，它们的意义相反。需要转写。  |
| dropout   | dropout   | 表示 dropout 概率。  |
| bidirectional | direction    | PyTorch 表示是否进行双向 GRU，Paddle 使用字符串表示是双向 GRU（`bidirectional`）还是单向 GRU（`forward`）。 |
| device   | -   | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dtype   | -   | Tensor 的所需数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数， Paddle 保持默认即可。  |


### 转写示例
#### bias：是否使用偏置
```python
# PyTorch 写法
torch.nn.GRU(16, 32, bias=True)

# Paddle 写法
paddle.nn.GRU(16, 32)
```
```python
# PyTorch 写法
torch.nn.GRU(16, 32, bias=False)

# Paddle 写法
paddle.nn.GRU(16, 32, bias_ih_attr=False, bias_hh_attr=False)
```

#### batch_first：batch size 是否为第一维
```python
# PyTorch 写法
torch.nn.GRU(16, 32, batch_first=True)

# Paddle 写法
paddle.nn.GRU(16, 32, time_major=False)
```

#### bidirectional：是否进行双向
```python
# PyTorch 写法
torch.nn.GRU(16, 32, bidirectional=True)

# Paddle 写法
paddle.nn.GRU(16, 32, direction='bidirectional')
```
```python
# PyTorch 写法
torch.nn.GRU(16, 32, bidirectional=False)

# Paddle 写法
paddle.nn.GRU(16, 32, direction='forward')
```
