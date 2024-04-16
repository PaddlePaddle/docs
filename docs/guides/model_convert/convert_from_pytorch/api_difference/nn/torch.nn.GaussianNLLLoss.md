## [仅参数名不一致]torch.nn.GaussianNLLLoss

### [torch.nn.GaussianNLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss)

```python
torch.nn.GaussianNLLLoss(*, full=False, eps=1e-06, reduction='mean')
```

### [paddle.nn.GaussianNLLLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/GaussianNLLLoss_cn.html)

```python
paddle.nn.GaussianNLLLoss(full=False, epsilon=1e-6, reduction='mean', name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle       | 备注                                                                               |
| ------------------ | ------------------ | ---------------------------------------------------------------------------------- |
| full               | full               | 是否在损失计算中包括常数项。默认情况下为 False，表示忽略最后的常数项。                 |
| eps                | epsilon            | 一个很小的数字，用于限制 variance 的值，使其不会导致除 0 的出现。默认值为 1e-6，仅参数名不一致。       |
| reduction          | reduction          | 指定应用于输出结果的计算方式，可选值有 `none`、`mean` 和 `sum`。默认为 `mean`，计算 mini-batch loss 均值。设置为 `sum` 时，计算 mini-batch loss 的总和。设置为 `none` 时，则返回 loss Tensor。默认值下为 `mean`。   |
