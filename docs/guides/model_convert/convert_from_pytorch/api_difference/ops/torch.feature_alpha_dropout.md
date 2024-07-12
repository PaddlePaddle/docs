## [ 仅参数名不一致 ]torch.feature_alpha_dropout

### [torch.feature\_alpha\_dropout](https://pytorch.org/docs/stable/jit_builtin_functions.html)

```python
torch.feature_alpha_dropout(input, p, training)
```

### [paddle.nn.functional.feature\_alpha\_dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/feature_alpha_dropout_cn.html#feature_alpha-dropout)

```python
paddle.nn.functional.feature_alpha_dropout(x, p=0.5, training=True, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注 |
| -------- | ------------ | -- |
| input    | x            | 输入的多维 Tensor，仅参数名不一致。 |
| p        | p            | 将输入节点置 0 的概率。 |
| training | training     | 标记是否为训练阶段。 |
