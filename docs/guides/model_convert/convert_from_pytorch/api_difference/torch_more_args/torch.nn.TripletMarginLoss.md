## [ torch 参数更多 ]torch.nn.TripletMarginLoss
### [torch.nn.TripletMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss)
```python
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.TripletMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/TripletMarginLoss_cn.html)
```python
paddle.nn.TripletMarginLoss(margin=1.0, p=2., epsilon=1e-6, swap=False, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| margin       | margin       | 手动指定间距。                                 |
| p            | p            | 手动指定范数。                                 |
| eps          | epsilon      | 防止除数为 0，仅参数名不一致。                 |
| swap         | swap         | 默认为 False。                                 |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。需要转写。       |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。需要转写。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

### 转写示例

#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(size_average=True)

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(size_average=False)

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(size_average=False)

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(reduce=False)

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(reduction='none')

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(reduction='mean')

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.TripletMarginLoss(reduction='sum')

# Paddle 写法
paddle.nn.TripletMarginLoss(reduction='sum')
```
