## [ 输入参数类型不一致 ]torch.multiply

### [torch.multiply](https://pytorch.org/docs/stable/generated/torch.multiply.html?highlight=torch+multiply#torch.multiply)

```python
torch.multiply(input, other, *, out=None)
```

### [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multiply_cn.html)

```python
paddle.multiply(x, y, name=None)
```

其中 PyTorch 和 Paddle 的 `other` 参数支持类型不一致，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| other     | y           | 表示输入的 Tensor ，torch 支持 Tensor 和 Python Number，paddle 仅支持 Tensor。当输入为 Python Number 时，需要转写。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要转写。         |

### 转写示例
#### other：输入为 Number
```python
# PyTorch 写法
torch.multiply(torch.tensor([2, 3, 8, 7]), other=2.0)

# Paddle 写法
paddle.multiply(paddle.to_tensor([2, 3, 8, 7]), other=paddle.to_tensor(2.0))
```

#### out：指定输出
```python
# PyTorch 写法
torch.multiply([3, 5], [1, 2], out = y)

# Paddle 写法
paddle.assign(paddle.multiply([3, 5], [1, 2]) , y)
```
