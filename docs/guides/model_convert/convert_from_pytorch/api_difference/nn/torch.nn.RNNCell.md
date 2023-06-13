## [torch 参数更多]torch.nn.RNNCell

### [torch.nn.RNNCell](https://pytorch.org/docs/1.13/generated/torch.nn.RNNCell.html#torch.nn.RNNCell)

```python
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
```

### [paddle.nn.SimpleRNNCell](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SimpleRNNCell_cn.html)

```python
paddle.nn.SimpleRNNCell(input_size, hidden_size, activation='tanh', weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle               | 备注                                                                           |
| ------------ | -------------------------- | ------------------------------------------------------------------------------ |
| input_size   | input_size                 | 输入的大小。                                                                   |
| hidden_size  | hidden_size                | 隐藏状态大小。                                                                 |
| bias         | bias_ih_attr, bias_hh_attr | 是否使用 blas 层权重，Paddle 使用 bias_ih_attr 和 bias_hh_attr，需要进行转写。 |
| nonlinearity | activation                 | 简单循环神经网络单元的激活函数，仅参数名不一致。                               |
| device       | -                          | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。                      |
| dtype        | -                          | 参数类型，Paddle 无此参数，需要进行转写。                                      |
| -            | weight_ih_attr             | weight_ih 的参数，PyTorch 无此参数，Paddle 保持默认即可。                      |
| -            | weight_hh_attr             | weight_hh 的参数，PyTorch 无此参数，Paddle 保持默认即可。                      |

### 转写示例

#### bias：是否使用 bias

```python
# Pytorch 写法
m1 = torch.nn.RNNCell(input_size, hidden_size，bias=True)
m2 = torch.nn.RNNCell(input_size, hidden_size，bias=False)

# Paddle 写法
m1 = paddle.nn.SimpleRNNCell(input_size, hidden_size,
                             bias_ih_attr=paddle.ParamAttr(learning_rate=0.0),
                             bias_hh_attr=paddle.ParamAttr(learning_rate=0.0))
m2 = paddle.nn.SimpleRNNCell(input_size, hidden_size, bias_ih_attr=None, bias_hh_attr=None)
```

#### device 参数：表示 Tensor 存放设备位置

```python
# PyTorch 写法:
torch.nn.RNNCell(input_size, hidden_size, device=torch.device('cpu'))

# Paddle 写法:
y = paddle.nn.SimpleRNNCell(input_size, hidden_size)
y.cpu()
```

#### dtype 参数：参数类型

```python
# PyTorch 写法:
torch.nn.RNNCell(input_size, hidden_size, dtype=torch.float32)

# Paddle 写法:
paddle.nn.SimpleRNNCell(input_size, hidden_size).astype(paddle.float32)
```
