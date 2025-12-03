## [ torch 参数更多 ]torch.nn.NLLLoss
### [torch.nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html?highlight=nllloss#torch.nn.NLLLoss)
```python
torch.nn.NLLLoss(weight=None,
                 size_average=None,
                 ignore_index=- 100,
                 reduce=None,
                 reduction='mean')
```

### [paddle.nn.NLLLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/NLLLoss_cn.html#nllloss)
```python
paddle.nn.NLLLoss(weight=None,
                  ignore_index=- 100,
                  reduction='mean',
                  name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                         |
| ------------ | ------------ | -------------------------------------------- |
| weight       | weight       | 表示每个类别的权重。                         |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| ignore_index | ignore_index | 表示忽略的一个标签值。                       |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction    | reduction    | 表示应用于输出结果的计算方式。               |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=False, reduce=True)
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=False)

# Paddle 写法
paddle.nn.NLLLoss(weight=w, ignore_index=-100, reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=True, reduce=True)
torch.nn.NLLLoss(weight=w, ignore_index=-100, reduce=True)
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=True)
torch.nn.NLLLoss(weight=w, ignore_index=-100)

# Paddle 写法
paddle.nn.NLLLoss(weight=w, ignore_index=-100, reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=True, reduce=False)
torch.nn.NLLLoss(weight=w, ignore_index=-100, size_average=False, reduce=False)
torch.nn.NLLLoss(weight=w, ignore_index=-100, reduce=False)

# Paddle 写法
paddle.nn.NLLLoss(weight=w, ignore_index=-100, reduction='none')
```
