## [ 仅参数名不一致 ]torch.tan

### [torch.tan](https://pytorch.org/docs/stable/generated/torch.tan.html)

```python
torch.tan(input, *, out=None)
```

### [paddle.tan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tan_cn.html#tan)

```python
paddle.tan(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |

### 转写示例

#### out

```python
# PyTorch
torch.tan(x, out=y)

# Paddle
paddle.assign(paddle.tan(x), y)
```
