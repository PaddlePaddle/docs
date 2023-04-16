## [ torch 参数更多 ]torch.Tensor.ger

### [torch.Tensor.ger](https://pytorch.org/docs/1.13/generated/torch.Tensor.ger.html?highlight=torch+tensor+ger#torch.Tensor.ger)

```python
torch.Tensor.ger(input, vec2, *, out=None)
```

### [paddle.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html)

```python
paddle.outer(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> vec2 </font>  | <font color='red'> y </font> | 输入的 Tensor ，仅参数名不同
| <font color='red'> out </font>   | -                            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
v1 = torch.arange(1., 5.)
v2 = torch.arange(1., 4.)
torch.Tensor.ger(v1, v2, out = out) # 同 out = torch.Tensor.ger(v1, v2)

# Paddle 写法
v1 = paddle.arange(1, 5).astype('float32')
v2 = paddle.arange(1, 4).astype('float32')
out = paddle.outer(v1, v2)
