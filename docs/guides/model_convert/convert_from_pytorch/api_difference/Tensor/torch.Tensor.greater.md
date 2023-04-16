## [ torch 参数更多 ]torch.Tensor.greater

### [torch.Tensor.greater](https://pytorch.org/docs/1.13/generated/torch.Tensor.greater.html?highlight=torch+tensor+greater#torch.Tensor.greater)

```python
torch.Tensor.greater(input, other, *, out=None)
```

### [paddle.greater_than](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/greater_than_cn.html)

```python
paddle.greater_than(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> other </font> | <font color='red'> y </font> | 输入的 Tensor ，仅参数名不同
| <font color='red'> out </font>   | -                            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.greater(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]), out = y) # 同 y = torch.Tensor.greater(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

# Paddle 写法
y = paddle.greater_than(paddle.to_tensor([[1, 2], [3, 4]]), paddle.to_tensor([[1, 1], [4, 4]]))
```
