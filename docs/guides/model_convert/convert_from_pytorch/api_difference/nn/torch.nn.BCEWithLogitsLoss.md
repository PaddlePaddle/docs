## [ torch 参数更多 ]torch.nn.BCEWithLogitsLoss
### [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#bcewithlogitsloss)

```python
torch.nn.BCEWithLogitsLoss(weight=None,
                           size_average=None,
                           reduce=None,
                           reduction='mean',
                           pos_weight=None)
```

### [paddle.nn.BCEWithLogitsLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/BCEWithLogitsLoss_cn.html#bcewithlogitsloss)

```python
paddle.nn.BCEWithLogitsLoss(weight=None,
                            reduction='mean',
                            pos_weight=None,
                            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| weight           | weight      | 表示每个 batch 二值交叉熵的权重。                                     |
| size_average  | -            | PyTorch 已弃用，paddle 需要转写。 |
| reduce        | -            | PyTorch 已弃用，paddle 需要转写。 |
| reduction  | reduction            | 表示应用于输出结果的计算方式。  |
| pos_weight  | pos_weight            | 表示正类的权重。  |

### 转写示例
#### size_average
size_average 为 True
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, size_average=True)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='mean')

```

size_average 为 False
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, size_average=False)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='sum')
```

#### reduce
reduce 为 True
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, reduce=True)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='mean')
```

reduce 为 False
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, reduce=False)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='none')
```

#### reduction
reduction 为'none'
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, reduction='none')

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='none')
```

reduction 为'mean'
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, reduction='mean')

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='mean')
```

reduction 为'sum'
```python
# Pytorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, reduction='sum')

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='sum')
```
