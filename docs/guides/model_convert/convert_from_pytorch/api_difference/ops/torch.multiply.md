## [ 参数不一致 ]torch.multiply

### [torch.multiply](https://pytorch.org/docs/1.13/generated/torch.multiply.html?highlight=torch+multiply#torch.multiply)

```python
torch.multiply(input, other, *, out=None)
```

### [paddle.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html)

```python
paddle.multiply(x, y, name=None)
```

其中 Pytorch 和 Paddle 的 `other` 参数支持类型不一致，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| other     | y           | 表示输入的 Tensor ，torch 支持 Tensor 和 Number，paddle 仅支持 Tensor。当输入为 Number 时，需要进行转写。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。         |

### 转写示例
#### other：输入为 Number
```python
# Pytorch 写法
torch.multiply(torch.tensor([2, 3, 8, 7]), other=2.0)

# Paddle 写法
paddle.multiply(paddle.to_tensor([2, 3, 8, 7]), other=paddle.to_tensor(2.0))
```

#### out：指定输出
```python
# Pytorch 写法
torch.multiply([3, 5], [1, 2], out = y)

# Paddle 写法
paddle.assign(paddle.multiply([3, 5], [1, 2]) , y)
```
