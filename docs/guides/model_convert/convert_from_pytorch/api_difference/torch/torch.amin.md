## [ torch 参数更多 ]torch.amin

### [torch.amin](https://pytorch.org/docs/stable/generated/torch.amin.html)

```python
torch.amin(input, dim, keepdim=False, *, out=None)
```

### [paddle.amin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amin_cn.html#amin)

```python
paddle.amin(x, axis=None, keepdim=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| dim     | axis         | 求最小值运算的维度，仅参数名不一致。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
| out     | -            | 输出 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out： 指定输出

```python
# PyTorch 写法
torch.amin(a, dim=0，out=y)

# Paddle 写法
paddle.assign(paddle.amin(a, dim=0), y)
```
