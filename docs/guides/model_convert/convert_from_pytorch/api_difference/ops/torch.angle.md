## [ 参数不一致 ]torch.angle

### [torch.angle](https://pytorch.org/docs/stable/generated/torch.angle.html)

```python
torch.angle(input, *, out=None)
```

### [paddle.angle](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/angle_cn.html#angle)

```python
paddle.angle(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入的 Tensor。 |
| out     | -            | 表示输出的 Tensor,可选项，Paddle 无此参数，需要转写。 |

### 转写示例

#### out： 指定输出

```python
# PyTorch 写法
torch.angle(x, out=y)

# Paddle 写法
paddle.assign(paddle.angle(x), y)
```
