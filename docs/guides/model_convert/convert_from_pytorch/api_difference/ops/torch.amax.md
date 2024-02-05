## [ torch 参数更多 ]torch.amax

### [torch.amax](https://pytorch.org/docs/stable/generated/torch.amax.html)

```python
torch.amax(input, dim, keepdim=False, *, out=None)
```

### [paddle.amax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amax_cn.html#amax)

```python
paddle.amax(x, axis=None, keepdim=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| dim     | axis         | 求最大值运算的维度，仅参数名不一致。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
| out     | -            | 输出 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out： 指定输出

```python
# PyTorch 写法
torch.amax(a, dim=0，out=y)

# Paddle 写法
paddle.assign(paddle.amax(a, dim=0), y)
```
