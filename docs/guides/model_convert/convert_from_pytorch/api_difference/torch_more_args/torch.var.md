## [ torch 参数更多 ]torch.var

### [torch.var](https://pytorch.org/docs/stable/generated/torch.var.html)

```python
torch.var(input, dim=None, unbiased=True, keepdim=False, *, correction=1, out=None)
```

### [paddle.var](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/var_cn.html#var)

```python
paddle.var(x, axis=None, unbiased=True, keepdim=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| input      | x            | 输入张量，仅参数名不一致。   |
| dim        | axis         | 指定对 x 进行计算的轴，仅参数名不一致。 |
| unbiased   | unbiased     | 是否使用无偏估计来计算标准差。 |
| keepdim    | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
| correction | -            | 样本尺寸与其自由度的差异，Paddle 无此参数，需要转写。   |
| out        | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |

### 转写示例

#### correction

```python
# PyTorch
torch.var(x, dim, correction=0)

# Paddle
paddle.var(x, dim, unbiased=False)


# PyTorch
torch.var(x, dim, correction=1)

# Paddle
paddle.var(x, dim, unbiased=True)
```

#### out

```python
# PyTorch
torch.var(x, out=y)

# Paddle
paddle.assign(paddle.var(x), y)
```
