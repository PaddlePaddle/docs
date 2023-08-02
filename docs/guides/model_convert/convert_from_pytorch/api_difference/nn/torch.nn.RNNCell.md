## [ 参数不一致 ]torch.nn.RNNCell
### [torch.nn.RNNCell](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell)
```python
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
```

### [paddle.nn.SimpleRNNCell](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SimpleRNNCell_cn.html#simplernncell)
```python
paddle.nn.SimpleRNNCell(input_size, hidden_size, activation='tanh', weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持，需要转写。   |
| nonlinearity          | activation            | 表示激活函数类型，仅参数名不一致。  |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数， Paddle 保持默认即可。  |

### 转写示例
#### bias：是否使用偏置
```python
# Pytorch 写法
torch.nn.RNNCell(16, 32, bias=True)

# Paddle 写法
paddle.nn.SimpleRNNCell(16, 32)
```
```python
# Pytorch 写法
torch.nn.RNNCell(16, 32, bias=False)

# Paddle 写法
paddle.nn.SimpleRNNCell(16, 32, bias_ih_attr=False, bias_hh_attr=False)
```
