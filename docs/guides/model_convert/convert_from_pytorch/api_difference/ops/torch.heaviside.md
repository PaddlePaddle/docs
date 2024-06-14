## [ torch 参数更多 ]torch.heaviside
### [torch.heaviside](https://pytorch.org/docs/stable/generated/torch.heaviside.html#torch.heaviside)

```python
torch.heaviside(input, values, *, out=None)
```

### [paddle.heaviside](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/heaviside_cn.html#heaviside)

```python
paddle.heaviside(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> values </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.heaviside(x, y, out=z)

# Paddle 写法
z = paddle.heaviside(x, y)
```
