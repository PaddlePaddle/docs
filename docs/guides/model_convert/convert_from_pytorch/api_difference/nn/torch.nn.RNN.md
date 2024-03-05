## [ 参数不一致 ]torch.nn.RNN
### [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN)
```python
torch.nn.RNN(input_size,
             hidden_size,
             num_layers=1,
             nonlinearity='tanh',
             bias=True,
             batch_first=False,
             dropout=0,
             bidirectional=False)
```

### [paddle.nn.SimpleRNN](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SimpleRNN_cn.html#simplernn)
```python
paddle.nn.SimpleRNN(input_size, hidden_size, num_layers=1, activation='tanh', direction='forward', dropout=0., time_major=False, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| num_layers          | num_layers            | 表示循环网络的层数。  |
| nonlinearity          | activation            | 表示激活函数类型，仅参数名不一致。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持，需要转写。   |
| batch_first   | time_major   | PyTorch 表示 batch size 是否为第一维，PaddlePaddle 表示 time steps 是否为第一维，它们的意义相反。需要转写。  |
| dropout   | dropout   | 表示 dropout 概率。  |
| bidirectional | direction    | PyTorch 表示是否进行双向 RNN，Paddle 使用字符串表示是双向 RNN（`bidirectional`）还是单向 RNN（`forward`）。 |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数， Paddle 保持默认即可。  |


### 转写示例
#### bias：是否使用偏置
```python
# PyTorch 写法
torch.nn.RNN(16, 32, bias=True)

# Paddle 写法
paddle.nn.SimpleRNN(16, 32)
```
```python
# PyTorch 写法
torch.nn.RNN(16, 32, bias=False)

# Paddle 写法
paddle.nn.SimpleRNN(16, 32, bias_ih_attr=False, bias_hh_attr=False)
```

#### batch_first：batch size 是否为第一维
```python
# PyTorch 写法
torch.nn.RNN(16, 32, batch_first=True)

# Paddle 写法
paddle.nn.SimpleRNN(16, 32, time_major=False)
```

#### bidirectional：是否进行双向
```python
# PyTorch 写法
torch.nn.RNN(16, 32, bidirectional=True)

# Paddle 写法
paddle.nn.RNN(16, 32, direction='bidirectional')
```
```python
# PyTorch 写法
torch.nn.RNN(16, 32, bidirectional=False)

# Paddle 写法
paddle.nn.RNN(16, 32, direction='forward')
```
