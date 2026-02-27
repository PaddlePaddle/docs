## [ torch 参数更多 ]torch.nn.MultiLabelMarginLoss
### [torch.nn.MultiLabelMarginLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss)
```python
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.MultiLabelMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MultiLabelMarginLoss_cn.html#paddle.nn.MultiLabelMarginLoss)
```python
paddle.nn.MultiLabelMarginLoss(reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式，需要转写。       |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式，需要转写。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.MultiLabelMarginLoss(size_average=False, reduce=True)
torch.nn.MultiLabelMarginLoss(size_average=False)

# Paddle 写法
paddle.nn.MultiLabelMarginLoss(reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.MultiLabelMarginLoss(size_average=True, reduce=True)
torch.nn.MultiLabelMarginLoss(reduce=True)
torch.nn.MultiLabelMarginLoss(size_average=True)
torch.nn.MultiLabelMarginLoss()

# Paddle 写法
paddle.nn.MultiLabelMarginLoss(reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.MultiLabelMarginLoss(size_average=True, reduce=False)
torch.nn.MultiLabelMarginLoss(size_average=False, reduce=False)
torch.nn.MultiLabelMarginLoss(reduce=False)

# Paddle 写法
paddle.nn.MultiLabelMarginLoss(reduction='none')
```
