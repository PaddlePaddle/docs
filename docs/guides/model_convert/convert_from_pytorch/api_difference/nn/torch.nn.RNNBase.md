## [torch 参数更多]torch.nn.RNNBase

### [torch.nn.RNNBase](https://pytorch.org/docs/1.13/generated/torch.nn.RNNBase.html#torch.nn.RNNBase)

```python
torch.nn.RNNBase(mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
```

### [paddle.nn.SimpleRNN](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SimpleRNN_cn.html)

```python
paddle.nn.SimpleRNN(input_size, hidden_size, num_layers=1, activation='tanh', direction='forward', dropout=0., time_major=False, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle               | 备注                                                                                                      |
| ------------- | -------------------------- | --------------------------------------------------------------------------------------------------------- |
| mode          | -                          | RNN 模式，Paddle 无此参数，暂无转写方式。                                                                 |
| input_size    | input_size                 | 输入 x 大小。                                                                                             |
| hidden_size   | hidden_size                | 隐藏状态 h 大小。                                                                                         |
| num_layers    | num_layers                 | 循环层数量。                                                                                              |
| bias          | bias_ih_attr, bias_hh_attr | 是否使用 bias ，Paddle 为 bias_ih，bias_hh 参数，需要进行转写。                                           |
| batch_first   | time_major                 | PyTorch 表示 batch size 是否为第一维，Paddle 表示 time steps 是否为第一维，它们的意义相反。需要进行转写。 |
| dropout       | dropout                    | 表示 dropout 概率。                                                                                       |
| bidirectional | direction                  | 是否双向 RNN，PyTorch 为 bool 类型，Paddle 为 forward 或 bidirect（或 bidirectional），需要进行转写。     |
| proj_size     | -                          | 投影大小，Paddle 无此参数，暂无转写方式。                                                                 |
| device        | -                          | Tensor 的设备，Paddle 无此参数，需要进行转写。                                                            |
| dtype         | -                          | Tensor 的数据类型，Paddle 无此参数，需要进行转写。                                                        |
| -             | activation                 | 激活函数类型，PyTorch 无此参数，Paddle 保持默认即可。                                                     |
| -             | weight_ih_attr             | weight_ih 的参数，PyTorch 无此参数，Paddle 保持默认即可。                                                 |
| -             | weight_hh_attr             | weight_hh 的参数，PyTorch 无此参数，Paddle 保持默认即可。                                                 |

### 转写示例

#### bias：是否使用 bias

```python
# Pytorch 写法
torch.nn.RNNBase('RNN_TANH', input_size, hidden_size, bias=True)

# Paddle 写法
paddle.nn.SimpleRNN(input_size, hidden_size)

# Pytorch 写法
torch.nn.RNNBase('RNN_TANH', input_size, hidden_size, bias=False)

# Paddle 写法
paddle.nn.SimpleRNN(input_size, hidden_size, bias_ih_attr=False, bias_hh_attr=False)
```

#### batch_first：batch size 是否为第一维

```python
# Pytorch 写法
torch.nn.RNNBase(input_size, hidden_size, batch_first=True)

# Paddle 写法
paddle.nn.SimpleRNN(input_size, hidden_size, time_major=False)
```

#### bidirectional：是否双向 RNN

```python
# Pytorch 写法
torch.nn.RNNBase('RNN_TANH', input_size, hidden_size, bidirectional=True)

# Paddle 写法
paddle.nn.SimpleRNN(input_size, hidden_size, direction='bidirectional')

# Pytorch 写法
torch.nn.RNNBase('RNN_TANH', input_size, hidden_size, bidirectional=False)

# Paddle 写法
paddle.nn.SimpleRNN(input_size, hidden_size, direction='forward')
```

#### device：Tensor 的设备

```python
# Pytorch 写法
m = torch.nn.RNNBase('RNN_TANH', input_size, hidden_size，device=torch.device('cpu'))
y = m(x)

# Paddle 写法
m = paddle.nn.SimpleRNN(input_size, hidden_size)
y = m(x).cpu()
```

#### dtype：Tensor 的数据类型

```python
# Pytorch 写法
m = torch.nn.RNNBase('RNN_TANH', input_size, hidden_size，dtype=torch.float32)
y = m(x)

# Paddle 写法
m = paddle.nn.SimpleRNN(input_size, hidden_size)
y = m(x).astype(paddle.float32)
```
