## [ torch 参数更多 ]torch.Tensor.heaviside

### [torch.Tensor.heaviside](https://pytorch.org/docs/1.13/generated/torch.Tensor.heaviside.html?highlight=torch+tensor+heaviside#torch.Tensor.heaviside)

```python
torch.Tensor.heaviside(input, values, *, out=None)
```

### [paddle.heaviside](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/heaviside_cn.html#heaviside)

```python
paddle.heaviside(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch                           | PaddlePaddle                 | 备注                                                   |
|-----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> input </font>  | <font color='red'> x </font> | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> values </font> | <font color='red'> y </font> | 输入的 Tensor ，仅参数名不同
| <font color='red'> out </font>    | -                            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
input = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
torch.Tensor.heaviside(input, values, out = out)

# Paddle 写法
input = paddle.to_tensor([-1.5, 0, 2.0])
values = paddle.to_tensor([0.5])
out = paddle.heaviside(input, values)
```
