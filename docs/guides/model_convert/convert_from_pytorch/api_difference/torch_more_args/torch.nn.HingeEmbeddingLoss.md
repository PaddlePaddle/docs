## [ torch 参数更多 ]torch.nn.HingeEmbeddingLoss
### [torch.nn.HingeEmbeddingLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss)
```python
torch.nn.HingeEmbeddingLoss(margin=1.0,
                            size_average=None,
                            reduce=None,
                            reduction='mean')
```

### [paddle.nn.HingeEmbeddingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/HingeEmbeddingLoss_cn.html#paddle.nn.HingeEmbeddingLoss)
```python
paddle.nn.HingeEmbeddingLoss(margin=1.0,
                             reduction='mean',
                             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| margin       | margin       | 当 label 为-1 时，该值决定了小于 margin 的 input 才需要纳入 hinge embedding loss 的计算。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                 |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                 |
| reduction    | reduction    | 表示应用于输出结果的计算方式。                               |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.HingeEmbeddingLoss(margin=m, size_average=False, reduce=True)
torch.nn.HingeEmbeddingLoss(margin=m, size_average=False)

# Paddle 写法
paddle.nn.HingeEmbeddingLoss(margin=m, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.HingeEmbeddingLoss(margin=m, size_average=True, reduce=True)
torch.nn.HingeEmbeddingLoss(margin=m, reduce=True)
torch.nn.HingeEmbeddingLoss(margin=m, size_average=True)
torch.nn.HingeEmbeddingLoss(margin=m)

# Paddle 写法
paddle.nn.HingeEmbeddingLoss(margin=m, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.HingeEmbeddingLoss(margin=m, size_average=True, reduce=False)
torch.nn.HingeEmbeddingLoss(margin=m, size_average=False, reduce=False)
torch.nn.HingeEmbeddingLoss(margin=m, reduce=False)

# Paddle 写法
paddle.nn.HingeEmbeddingLoss(margin=m, reduction='none')
```
