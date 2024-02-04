## [ 仅参数名不一致 ]torch.alpha_dropout

### [torch.alpha\_dropout](https://pytorch.org/docs/master/generated/torch.nn.functional.alpha_dropout.html)

```python
torch.alpha_dropout(input, p=0.5, train=False)
```

### [paddle.nn.functional.alpha\_dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/alpha_dropout_cn.html#alpha-dropout)

```python
paddle.nn.functional.alpha_dropout(x, p=0.5, training=True, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入的多维 Tensor，数据类型为：float16、float32 或 float64。 |
| p       | p            | 将输入节点置 0 的概率，即丢弃概率。默认：0.5。 |
| train   | training     | 标记是否为训练阶段。默认：True。 |
