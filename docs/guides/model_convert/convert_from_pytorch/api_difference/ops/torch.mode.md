## [ torch 参数更多 ]torch.mode

### [torch.mode](https://pytorch.org/docs/stable/generated/torch.mode.html)

```python
torch.mode(input, dim=-1, keepdim=False, *, out=None)
```

### [paddle.mode](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mode_cn.html#mode)

```python
paddle.mode(x, axis=-1, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入的多维 Tensor。 |
| dim     | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。 |
| keepdim | keepdim      | 是否保留指定的轴。 |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。     |

###  转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.mode(x, dim, False, out=(a, b))

# Paddle 写法
out1, out2 = paddle.mode(x, dim, False)
paddle.assign(out1, (a, b)[0]), paddle.assign(out2, (a, b)[1])
```
