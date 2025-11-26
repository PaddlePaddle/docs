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
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(margin=m, size_average=False, reduce=True)
torch.nn.CosineEmbeddingLoss(margin=m, size_average=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(margin=m, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(margin=m, size_average=True, reduce=True)
torch.nn.CosineEmbeddingLoss(margin=m, reduce=True)
torch.nn.CosineEmbeddingLoss(margin=m, size_average=True)
torch.nn.CosineEmbeddingLoss(margin=m)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(margin=m, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.CosineEmbeddingLoss(margin=m, size_average=True, reduce=False)
torch.nn.CosineEmbeddingLoss(margin=m, size_average=False, reduce=False)
torch.nn.CosineEmbeddingLoss(margin=m, reduce=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLoss(margin=m, reduction='none')
```
