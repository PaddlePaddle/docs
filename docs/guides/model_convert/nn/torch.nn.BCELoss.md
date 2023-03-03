# torch.nn.BCELoss
### [torch.nn.BCELoss](https://pytorch.org/docs/1.13/generated/torch.nn.BCELoss.html?highlight=bceloss#torch.nn.BCELoss)

```python
torch.nn.BCELoss(weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean')
```

### [paddle.nn.BCELoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BCELoss_cn.html#bceloss)

```python
paddle.nn.BCELoss(weight=None,
                  reduction='mean',
                  name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | 已弃用。  |
| reduce        | -            | 已弃用。  |

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
