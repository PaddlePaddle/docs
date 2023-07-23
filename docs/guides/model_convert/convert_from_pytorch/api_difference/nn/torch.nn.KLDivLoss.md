# [torch 参数更多 ]torch.nn.KLDivLoss
### [torch.nn.KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss)

```python
torch.nn.KLDivLoss(size_average=None,
                   reduce=None,
                   reduction='mean',
                   log_target=False)
```

### [paddle.nn.KLDivLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/KLDivLoss_cn.html#kldivloss)

```python
paddle.nn.KLDivLoss(reduction='mean')
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。            |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示对输出结果的计算方式。  |
| log_target    | -            | 指定目标是否为日志空间，Paddle 无此功能，暂无转写方式。  |

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
