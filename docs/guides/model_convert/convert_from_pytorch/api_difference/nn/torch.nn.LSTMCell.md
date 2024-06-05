## [ 参数不一致 ]torch.nn.LSTMCell
### [torch.nn.LSTMCell](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell)
```python
torch.nn.LSTMCell(input_size, hidden_size, bias=True, device=None, dtype=None)
```

### [paddle.nn.LSTMCell](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LSTMCell_cn.html#lstmcell)
```python
paddle.nn.LSTMCell(input_size, hidden_size, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, proj_size=0, name=None)
```

两者功能一致但参数不一，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置， Paddle 支持自定义偏置属性， torch 不支持，需要转写。   |
| device   | -   | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dtype   | -   | Tensor 的所需数据类型，一般对网络训练结果影响不大，可直接删除。  |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |proj_size|表示是否将 `hidden state` 映射到对应的大小， PyTorch 无此参数， Paddle 保持默认即可。  |


### 转写示例
#### bias：是否使用偏置
```python
# PyTorch 写法
torch.nn.LSTMCell(16, 32, bias=True)

# Paddle 写法
paddle.nn.LSTMCell(16, 32)
```
```python
# PyTorch 写法
torch.nn.LSTMCell(16, 32, bias=False)

# Paddle 写法
paddle.nn.LSTMCell(16, 32, bias_ih_attr=False, bias_hh_attr=False)
```
