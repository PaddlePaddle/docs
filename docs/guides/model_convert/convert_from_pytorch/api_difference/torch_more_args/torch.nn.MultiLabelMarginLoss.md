## [torch 参数更多]torch.nn.MultiLabelMarginLoss

### [torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss)

```python
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.MultiLabelMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MultiLabelMarginLoss_cn.html)

```python
paddle.nn.MultiLabelMarginLoss(reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。       |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

### 转写示例

```python
# PyTorch 的 size_average、reduce 参数转为 Paddle 的 reduction 参数
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
```
