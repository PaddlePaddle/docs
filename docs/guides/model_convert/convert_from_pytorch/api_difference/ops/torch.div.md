## [ torch 参数更多 ]torch.div
### [torch.div](https://pytorch.org/docs/stable/generated/torch.div.html#torch.div)
```python
torch.div(input,
          other,
          *,
          rounding_mode=None,
          out=None)
```

### [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/divide_cn.html)
```python
paddle.divide(x,
              y,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
| other |  y  | 表示输入的 Tensor，仅参数名不一致。  |
| rounding_mode  | -  | 表示舍入模式，Paddle 无此参数, 需要进行转写。  |
|  out  | -   | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。    |


### 转写示例
#### rounding_mode: 舍入模式
```python
# Pytorch 写法 (rounding_mode 参数设置为 None)
x = torch.div(torch.tensor([2, 3, 4]), torch.tensor([1, 5, 2]))

# Paddle 写法
a = paddle.to_tensor([2, 3, 4], dtype='float64')
b = paddle.to_tensor([1, 5, 2], dtype='float64')
x = paddle.divide(a, b)

# Pytorch 写法 (rounding_mode 参数设置为"trunc")
x = torch.div(torch.tensor([-0.3711, -1.9353]), torch.tensor([0.8032,  0.2930]), rounding_mode='trunc')

# Paddle 写法
x = paddle.divide(paddle.to_tensor([-0.3711, -1.9353]), paddle.to_tensor([0.8032,  0.2930]))
x = paddle.trunc(x)

# Pytorch 写法 (rounding_mode 参数设置为"floor")
x = torch.div(torch.tensor([-0.3711, -1.9353]), torch.tensor([0.8032,  0.2930]), rounding_mode='floor')

# Paddle 写法
x = paddle.divide(paddle.to_tensor([-0.3711, -1.9353]), paddle.to_tensor([0.8032,  0.2930]))
x = paddle.floor(x)
```

#### out：指定输出
```python
# Pytorch 写法
torch.div(torch.tensor([2, 3, 4]), torch.tensor([1, 5, 2]), out=y)

# Paddle 写法
paddle.assign(paddle.divide(paddle.to_tensor([2, 3, 4]), paddle.to_tensor([1, 5, 2])), y)
```
