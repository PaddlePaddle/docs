## [torch 参数更多]torch.nn.SoftMarginLoss

### [torch.nn.SoftMarginLoss](https://pytorch.org/docs/1.13/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss)

```python
torch.nn.SoftMarginLoss(size_average=None,
                        reduce=None,
                        reduction='mean')
```

### [paddle.nn.SoftMarginLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SoftMarginLoss_cn.html#softmarginloss)

```python
paddle.nn.SoftMarginloss(reduction='mean',
                         name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | Paddle    | 备注                                         |
| ------------ | --------- | -------------------------------------------- |
| size_average | -         | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce       | -         | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction    | reduction | 表示应用于输出结果的计算方式。               |

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
