## [ torch 参数更多 ]torch.nn.SoftMarginLoss
### [torch.nn.SoftMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss)
```python
torch.nn.SoftMarginLoss(size_average=None,
                        reduce=None,
                        reduction='mean')
```

### [paddle.nn.SoftMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SoftMarginLoss_cn.html#softmarginloss)
```python
paddle.nn.SoftMarginLoss(reduction='mean',
                         name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                         |
| ------------ | ------------ | -------------------------------------------- |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction    | reduction    | 表示应用于输出结果的计算方式。               |

### 转写示例

#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(size_average=True)

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(size_average=False)

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(size_average=False)

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(reduce=False)

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(reduction='none')

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(reduction='mean')

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.SoftMarginLoss(reduction='sum')

# Paddle 写法
paddle.nn.SoftMarginLoss(reduction='sum')
```
