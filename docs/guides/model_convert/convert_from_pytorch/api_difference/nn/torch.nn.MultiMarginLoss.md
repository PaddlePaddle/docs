## [torch 参数更多]torch.nn.MultiMarginLoss

### [torch.nn.MultiMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss)

```python
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.MultiMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MultiMarginLoss_cn.html)

```python
paddle.nn.MultiMarginLoss(p: int = 1, margin: float = 1.0, weight=None, reduction: str = 'mean', name: str = None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle       | 备注                                                                               |
| ------------------ | ------------------ | ---------------------------------------------------------------------------------- |
| p                  | p                  | 手动指定幂次方指数大小，默认为 1。                                                   |
| margin             | margin             | 手动指定间距，默认为 1。                                                             |
| weight             | weight             | 权重值，默认为 None。 如果给定权重则形状为 `[C,]`                                       |
| size_average       | -                  | 已废弃（可用 `reduction` 代替）。表示是否采用 batch 中各样本 loss 平均值作为最终的 loss。如果置 False，则采用加和作为 loss。默认为 True，paddle 需要转写。    |
| reduce             | -                  | 已废弃（可用 `reduction` 代替）。表示是否采用输出单个值作为 loss。如果置 False，则每个元素输出一个 loss 并忽略 `size_average`。默认为 True，paddle 需要转写。 |
| reduction          | reduction          | 指定应用于输出结果的计算方式，可选值有 `none`、`mean` 和 `sum`。默认为 `mean`，计算 mini-batch loss 均值。设置为 `sum` 时，计算 mini-batch loss 的总和。设置为 `none` 时，则返回 loss Tensor。默认值下为 `mean`。   |

### 转写示例

```python
# Pytorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
if size_average is None:
    size_average = True
if reduce is None:
    reduce = True
if size_average and reduce:
    reduction = 'mean'
elif reduce:
    reduction = 'sum'
else:
    reduction = 'none'

# 如果 Pytorch 存在 reduction 参数，则直接覆盖
if 'reduction' not in kwargs:
    kwargs['reduction'] = reduction
```
