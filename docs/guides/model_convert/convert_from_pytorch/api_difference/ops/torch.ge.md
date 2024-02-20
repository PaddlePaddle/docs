## [ 参数完全一致 ]torch.ge

### [torch.ge](https://pytorch.org/docs/stable/generated/torch.ge.html)

```python
torch.ge(input, other, *, out)
```

### [paddle.greater\_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/greater_equal_cn.html#greater-equal)

```python
paddle.greater_equal(x, y)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                         |
| ------- | ------------ | --------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| other   | y            | 输入 Tensor，仅参数名不一致。 |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |

### 转写示例

#### out 参数：指定输出
``` python
# PyTorch 写法:
torch.ge(x, y, out=out)

# Paddle 写法:
paddle.assign(paddle.greater_equal(x, y), out)
```
