## [ 返回参数类型不一致 ]torch.nn.GRUCell

### [torch.nn.GRUCell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell)
```python
torch.nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
```

### [paddle.nn.GRUCell](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GRUCell_cn.html#grucell)
```python
paddle.nn.GRUCell(input_size, hidden_size, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)
```

两者功能一致但输入参数用法不一致，且返回参数类型不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input_size          | input_size            | 表示输入 x 的大小。  |
| hidden_size          | hidden_size            | 表示隐藏状态 h 大小。  |
| bias          | bias_ih_attr, bias_hh_attr  | 是否使用偏置，Paddle 的 bias_ih_attr, bias_hh_attr 参数均需与 PyTorch 设置一致，需要转写。   |
| device   | -   | 指定 Tensor 的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dtype   | -   | Tensor 的所需数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -             |weight_ih_attr| weight_ih 的参数， PyTorch 无此参数， Paddle 保持默认即可。  |
| -             |weight_hh_attr| weight_hh 的参数，  PyTorch 无此参数， Paddle 保持默认即可。  |


### 转写示例
#### bias：是否使用偏置
```python
# PyTorch 写法
torch.nn.GRUCell(16, 32, bias=True)
# Paddle 写法
paddle.nn.GRUCell(16, 32)
```
```python
# PyTorch 写法
torch.nn.GRUCell(16, 32, bias=False)
# Paddle 写法
paddle.nn.GRUCell(16, 32, bias_ih_attr=False, bias_hh_attr=False)
```
#### forward 类方法：前向传播
```python
# PyTorch 写法
rnn = torch.nn.GRUCell(2, 2)
result = rnn(inp, h0)

# Paddle 写法
rnn = paddle.nn.GRUCell(2, 2)
result = rnn(inp, h0)[0]
```
