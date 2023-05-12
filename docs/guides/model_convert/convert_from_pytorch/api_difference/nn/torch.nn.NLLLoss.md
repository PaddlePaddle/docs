## [torch 参数更多]torch.nn.NLLLoss

### [torch.nn.NLLLoss](https://pytorch.org/docs/1.13/generated/torch.nn.NLLLoss.html?highlight=nllloss#torch.nn.NLLLoss)

```python
torch.nn.NLLLoss(weight=None,
                 size_average=None,
                 ignore_index=- 100,
                 reduce=None,
                 reduction='mean')
```

### [paddle.nn.NLLLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/NLLLoss_cn.html#nllloss)

```python
paddle.nn.NLLLoss(weight=None,
                  ignore_index=- 100,
                  reduction='mean',
                  name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch      | Paddle       | 备注                                         |
| ------------ | ------------ | -------------------------------------------- |
| weight       | weight       | 表示每个类别的权重。                         |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| ignore_index | ignore_index | 表示忽略的一个标签值。                       |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction    | reduction    | 表示应用于输出结果的计算方式。               |

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
