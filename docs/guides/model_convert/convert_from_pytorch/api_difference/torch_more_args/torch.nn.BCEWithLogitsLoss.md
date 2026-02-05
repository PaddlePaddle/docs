## [ torch 参数更多 ]torch.nn.BCEWithLogitsLoss
### [torch.nn.BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)
```python
torch.nn.BCEWithLogitsLoss(weight=None,
                           size_average=None,
                           reduce=None,
                           reduction='mean',
                           pos_weight=None)
```

### [paddle.nn.BCEWithLogitsLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/BCEWithLogitsLoss_cn.html#paddle.nn.BCEWithLogitsLoss)
```python
paddle.nn.BCEWithLogitsLoss(weight=None,
                            reduction='mean',
                            pos_weight=None,
                            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| weight           | weight      | 表示每个 batch 二值交叉熵的权重。                                     |
| size_average  | -            | PyTorch 已弃用，paddle 需要转写。 |
| reduce        | -            | PyTorch 已弃用，paddle 需要转写。 |
| reduction  | reduction            | 表示应用于输出结果的计算方式。  |
| pos_weight  | pos_weight            | 表示正类的权重。  |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, size_average=False, reduce=True)
torch.nn.BCEWithLogitsLoss(weight=w, size_average=False)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, size_average=True, reduce=True)
torch.nn.BCEWithLogitsLoss(weight=w, reduce=True)
torch.nn.BCEWithLogitsLoss(weight=w, size_average=True)
torch.nn.BCEWithLogitsLoss(weight=w)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.BCEWithLogitsLoss(weight=w, size_average=True, reduce=False)
torch.nn.BCEWithLogitsLoss(weight=w, size_average=False, reduce=False)
torch.nn.BCEWithLogitsLoss(weight=w, reduce=False)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(weight=w, reduction='none')
```
