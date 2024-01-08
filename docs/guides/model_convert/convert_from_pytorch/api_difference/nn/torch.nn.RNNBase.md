## [torch 参数更多] torch.nn.RNNBase
### [torch.nn.RNNBase](https://pytorch.org/docs/stable/generated/torch.nn.RNNBase.html#torch.nn.RNNBase)
```python
torch.nn.RNNBase(mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None)
```

### [paddle.nn.layer.rnn.RNNBase](https://github.com/PaddlePaddle/Paddle/blob/e25e86f4f6d1bbd043b621a75e93d0070719c3d8/python/paddle/nn/layer/rnn.py#L1300)
```python
paddle.nn.layer.rnn.RNNBase(mode, input_size, hidden_size,
        num_layers=1, direction="forward", time_major=False,
        dropout=0.0, weight_ih_attr=None, weight_hh_attr=None,
        bias_ih_attr=None, bias_hh_attr=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| mode          | mode         | 表示 `RNN` 模型的类型,torch 取值为 `'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU`，paddle 取值为 `'LSTM', 'GRU'`，需要转写。|
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| num_layers          | num_layers            | 表示循环网络的层数。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持，需要转写。   |
| batch_first   | time_major   | PyTorch 表示 batch size 是否为第一维，PaddlePaddle 表示 time steps 是否为第一维，它们的意义相反。需要转写。  |
| dropout   | dropout   | 表示 dropout 概率。  |
| bidirectional | direction    | PyTorch 表示是否进行双向 RNN，Paddle 使用字符串表示是双向 RNN（`bidirectional`）还是单向 RNN（`forward`）。 |
| proj_size | - | PyTorch 表示投影层大小，默认为 0，表示不使用投影。Paddle 无此参数，一般对网络训练结果影响不大，现阶段直接删除。 |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数，Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数，Paddle 保持默认即可。  |

### 转写示例

#### mode：RNN 模型类别
```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32)
```

```python
# PyTorch 写法
torch.nn.RNNBase('GRU', 16, 32)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('GRU', 16, 32)
```

```python
# PyTorch 写法
torch.nn.RNNBase('RNN_TANH', 16, 32)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('SimpleRNN', 16, 32)
```

#### bias：是否使用偏置
```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32, bias=True)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32)
```

```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32, bias=False)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32, bias_ih_attr=False, bias_hh_attr=False)
```

#### batch_first：batch size 是否为第一维
```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32, batch_first=True)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32, time_major=False)
```

#### bidirectional：是否进行双向
```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32, bidirectional=True)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32, direction='bidirectional')
```
```python
# PyTorch 写法
torch.nn.RNNBase('LSTM', 16, 32, bidirectional=False)

# Paddle 写法
paddle.nn.layer.rnn.RNNBase('LSTM', 16, 32, direction='forward')
```
