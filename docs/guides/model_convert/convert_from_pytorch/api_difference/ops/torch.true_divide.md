## [ torch 参数更多 ]torch.true_divide

### [torch.true\_divide](https://pytorch.org/docs/stable/generated/torch.true_divide.html)

```python
torch.true_divide(input, other, *, out)
```

### [paddle.divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/divide_cn.html#divide)

```python
paddle.divide(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| other   | y            | 输入 Tensor，仅参数名不一致。 |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |

### 转写示例

#### out

```python
# PyTorch
torch.true_divide(x, y, out=z)

# Paddle
paddle.assign(paddle.div(x, y), z)
```
