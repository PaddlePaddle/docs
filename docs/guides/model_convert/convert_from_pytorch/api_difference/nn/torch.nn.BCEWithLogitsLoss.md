# [ torch 参数更多 ]torch.nn.BCEWithLogitsLoss
### [torch.nn.BCEWithLogitsLoss](https)

```python
torch.nn.BCEWithLogitsLoss(weight=None,
                           size_average=None,
                           reduce=None,
                           reduction='mean',
                           pos_weight=None)
```

### [paddle.nn.BCEWithLogitsLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BCEWithLogitsLoss_cn.html#bcewithlogitsloss)

```python
paddle.nn.BCEWithLogitsLoss(weight=None,
                            reduction='mean',
                            pos_weight=None,
                            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| weight           | weight      | 表示每个 batch 二值交叉熵的权重。                                     |
| size_average  | -            | PyTorch 已弃用。  |
| reduce        | -            | PyTorch 已弃用。  |
| reduction  | reduction            | 表示应用于输出结果的计算方式。  |
| pos_weight  | pos_weight            | 表示正类的权重。  |

### 转写示例
#### size_average
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
