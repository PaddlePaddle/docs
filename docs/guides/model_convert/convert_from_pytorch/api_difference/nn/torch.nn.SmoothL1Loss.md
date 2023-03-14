# [torch 参数更多 ]torch.nn.SmoothL1Loss
### [torch.nn.SmoothL1Loss](https://pytorch.org/docs/1.13/generated/torch.nn.SmoothL1Loss.html?highlight=smoothl1loss#torch.nn.SmoothL1Loss)

```python
torch.nn.SmoothL1Loss(size_average=None,
                      reduce=None,
                      reduction='mean',
                      beta=1.0)
```

### [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SmoothL1Loss_cn.html#smoothl1loss)

```python
paddle.nn.SmoothL1Loss(reduction='mean',
                       delta=1.0,
                       name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | 已弃用。  |
| reduce        | -            | 已弃用。  |
| reduction        | reduction            | 表示对输出 Tensor 的计算方式。  |
| beta          | delta        | SmoothL1Loss 损失的阈值参数。  |

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
