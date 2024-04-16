## [ torch 参数更多]torch.renorm

### [torch.renorm](https://pytorch.org/docs/stable/generated/torch.renorm.html#torch-renorm)

```python
torch.renorm(input, p, dim, maxnorm, *, out=None)
```

### [paddle.renorm]()

```python
paddle.renorm(input, p, axis, max_norm)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| input   | input        | 表示输入的 Tensor 。                                |
| p       | p            | 表示 p-范数计算的 p 值。|
| dim     | axis         | 表示切分的维度，仅参数名不一致。                                    |
| maxnorm | max_norm     | 表示子张量的 p-范数最大值，仅参数名不一致。          |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |
### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.renorm(x, 1, 0, 5, out=output)

# Paddle 写法
paddle.assign(paddle.renorm(x, 1, 0, 5), output)
```
