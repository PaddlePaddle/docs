## [ torch 参数更多 ]torch.nn.MultiLabelSoftMarginLoss
### [torch.nn.MultiLabelSoftMarginLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss)
```python
torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.MultiLabelSoftMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MultiLabelSoftMarginLoss_cn.html#paddle.nn.MultiLabelSoftMarginLoss)
```python
paddle.nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| weight       | weight       | 手动设定权重。                                 |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。需要转写。       |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。需要转写。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=False, reduce=True)
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=False)

# Paddle 写法
paddle.nn.MultiLabelSoftMarginLoss(weight=w, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=True, reduce=True)
torch.nn.MultiLabelSoftMarginLoss(weight=w, reduce=True)
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=True)
torch.nn.MultiLabelSoftMarginLoss(weight=w)

# Paddle 写法
paddle.nn.MultiLabelSoftMarginLoss(weight=w, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=True, reduce=False)
torch.nn.MultiLabelSoftMarginLoss(weight=w, size_average=False, reduce=False)
torch.nn.MultiLabelSoftMarginLoss(weight=w, reduce=False)

# Paddle 写法
paddle.nn.MultiLabelSoftMarginLoss(weight=w, reduction='none')
```
