## [ torch 参数更多 ]torch.nn.PoissonNLLLoss
### [torch.nn.PoissonNLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss)
```python
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

### [paddle.nn.PoissonNLLLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/PoissonNLLLoss_cn.html#paddle.nn.PoissonNLLLoss)
```python
paddle.nn.PoissonNLLLoss(log_input=True, full=False, epsilon=1e-8, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle       | 备注                                                                               |
| ------------------ | ------------------ | ---------------------------------------------------------------------------------- |
| log_input          | log_input          | 输入是否为对数函数映射后结果。                                                       |
| full               | full               | 是否在损失计算中包括 Stirling 近似项。                                               |
| size_average       | -                  | 已废弃（可用 `reduction` 代替）。表示是否采用 batch 中各样本 loss 平均值作为最终的 loss。如果置 False，则采用加和作为 loss。默认为 True，paddle 需要转写。    |
| eps                | epsilon                | 在 `log_input` 为 True 时使用的常数小量。默认值为 1e-8，仅参数名不一致。                            |
| reduce             | -                  | 已废弃（可用 `reduction` 代替）。表示是否采用输出单个值作为 loss。如果置 False，则每个元素输出一个 loss 并忽略 `size_average`。默认为 True，paddle 需要转写。 |
| reduction          | reduction          | 指定应用于输出结果的计算方式，可选值有 `none`、`mean` 和 `sum`。默认为 `mean`，计算 mini-batch loss 均值。设置为 `sum` 时，计算 mini-batch loss 的总和。设置为 `none` 时，则返回 loss Tensor。默认值下为 `mean`。两者完全一致。   |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=False, reduce=True)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=False)

# Paddle 写法
paddle.nn.PoissonNLLLoss(log_input=True, full=False, epsilon=1e-8, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=True, reduce=True)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, reduce=True)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=True)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8)

# Paddle 写法
paddle.nn.PoissonNLLLoss(log_input=True, full=False, epsilon=1e-8, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=True, reduce=False)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, size_average=False, reduce=False)
torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, reduce=False)

# Paddle 写法
paddle.nn.PoissonNLLLoss(log_input=True, full=False, epsilon=1e-8, reduction='none')
```
