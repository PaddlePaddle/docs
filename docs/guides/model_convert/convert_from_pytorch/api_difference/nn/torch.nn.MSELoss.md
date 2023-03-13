## [torch 参数更多 ]torch.nn.MSELoss
### [torch.nn.MSELoss](https://pytorch.org/docs/1.13/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss)

```python
torch.nn.MSELoss(size_average=None,
                 reduce=None,
                 reduction='mean')
```

### [paddle.nn.MSELoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MSELoss_cn.html#mseloss)

```python
paddle.nn.MSELoss(reduction='mean')
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | 已弃用。  |
| reduce        | -            | 已弃用。  |
| reduction        | reduction            | 表示对输出结果的计算方式。  |

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
