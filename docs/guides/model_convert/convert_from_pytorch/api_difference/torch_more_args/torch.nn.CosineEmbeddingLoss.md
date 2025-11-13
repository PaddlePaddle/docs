## [ torch 参数更多 ]torch.nn.CosineEmbeddingLoss
### [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss)
```python
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.CosineEmbeddingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/CosineEmbeddingLoss_cn.html)
```python
paddle.nn.CosineEmbeddingLoss(margin=0, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| margin       | margin       | 可以设置的范围为[-1, 1]。                      |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。 需要转写。      |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。需要转写。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(size_average=True)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(size_average=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(reduce=True)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(reduce=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(reduction='none')

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(reduction='mean')

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(reduction='sum')

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(reduction='sum')
```
