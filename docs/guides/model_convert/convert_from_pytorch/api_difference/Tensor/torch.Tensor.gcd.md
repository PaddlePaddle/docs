## [ torch 参数更多 ]torch.Tensor.gcd.md

同 torch.gcd

### [torch.Tensor.gcd](https://pytorch.org/docs/1.13/generated/torch.Tensor.gcd.html?highlight=torch+tensor+gcd#torch.Tensor.gcd)

```python
torch.Tensor.gcd(input, other, *, out=None)
```

### [paddle.gcd(x, y, name=None)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/gcd_cn.html)

```python
paddle.gcd(x, y, name=None)
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
torch.Tensor.gcd([-3, -5], out=y) # 同 y = torch.Tensor.gcd([-3, -5])

# Paddle 写法
y = paddle.gcd([-3, -5])
```
