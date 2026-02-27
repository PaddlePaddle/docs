## [ torch 参数更多 ]torch.nn.functional.multilabel_soft_margin_loss
### [torch.nn.functional.multilabel\_soft\_margin\_loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.multilabel_soft_margin_loss.html#torch.nn.functional.multilabel_soft_margin_loss)
```python
torch.nn.functional.multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.multi\_label\_soft\_margin\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/multi_label_soft_margin_loss_cn.html#paddle.nn.functional.multi_label_soft_margin_loss)
```python
paddle.nn.functional.multi_label_soft_margin_loss(input, label, weight=None, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor。 |
| target       | label        | 标签 Tensor，仅参数名不一致。 |
| weight       | weight       | 手动指定每个 batch 二值交叉熵的权重。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(size_average=True)

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(reduce=True)

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(reduce=False)

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(reduction='none')

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(reduction='mean')

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.multilabel_soft_margin_loss(reduction='sum')

# Paddle 写法
paddle.nn.functional.multi\_label\_soft\_margin\_loss(reduction='sum')
```
