## [torch 参数更多]torch.nn.CosineEmbeddingLoss

### [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss)

```python
torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.CosineEmbeddingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/CosineEmbeddingLoss_cn.html)

```python
paddle.nn.CosineEmbeddingLoss(margin=0, reduction='mean', name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                           |
| ------------ | ------------ | ---------------------------------------------- |
| margin       | margin       | 可以设置的范围为[-1, 1]。                      |
| size_average | -            | 已废弃，和 reduce 组合决定损失计算方式。       |
| reduce       | -            | 已废弃，和 size_average 组合决定损失计算方式。 |
| reduction    | reduction    | 指定应用于输出结果的计算方式。                 |

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
```
