## [ torch 参数更多 ]torch.nn.MarginRankingLoss
### [torch.nn.MarginRankingLoss](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#marginrankingloss)
```python
torch.nn.MarginRankingLoss(margin=0.0,
                           size_average=None,
                           reduce=None,
                           reduction='mean')
```

### [paddle.nn.MarginRankingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MarginRankingLoss_cn.html#marginrankingloss)
```python
paddle.nn.MarginRankingLoss(margin=0.0,
                            reduction='mean',
                            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                         |
| ------------ | ------------ | -------------------------------------------- |
| margin       | margin       | 用于加和的 margin 值。                       |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction    | reduction    | 表示应用于输出结果的计算方式。               |

### 转写示例

#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(size_average=True)

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(size_average=False)

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(size_average=False)

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(reduce=False)

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(reduction='none')

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(reduction='mean')

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.MarginRankingLoss(reduction='sum')

# Paddle 写法
paddle.nn.MarginRankingLoss(reduction='sum')
```
