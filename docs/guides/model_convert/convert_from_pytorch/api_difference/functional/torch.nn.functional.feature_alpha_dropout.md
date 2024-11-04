## [ torch 参数更多 ]torch.nn.functional.feature_alpha_dropout

### [torch.nn.functional.feature\_alpha\_dropout](https://pytorch.org/docs/stable/generated/torch.nn.functional.feature_alpha_dropout.html)

```python
torch.nn.functional.feature_alpha_dropout(input, p=0.5, training=False, inplace=False)
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
| training | training     | 标记是否为训练阶段，参数默认值不一致，PyTorch 默认值为 False，paddle 默认值为 True，Paddle 需设置为与 PyTorch 一致。 |
| inplace  | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
