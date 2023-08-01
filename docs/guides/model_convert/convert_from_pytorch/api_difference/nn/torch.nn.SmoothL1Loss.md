# [torch 参数更多 ]torch.nn.SmoothL1Loss
### [torch.nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=smoothl1loss#torch.nn.SmoothL1Loss)

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

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示应用于输出结果的计算方式。 |
| beta          | delta        | SmoothL1Loss 损失的阈值参数。  |

### 转写示例
#### size_average
```python
# Paddle 写法
torch.nn.SmoothL1Loss(size_average=True)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

#### size_average
size_average 为 True
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(size_average=True)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')

```

size_average 为 False
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(size_average=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='sum')
```

#### reduce
reduce 为 True
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(reduce=True)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

reduce 为 False
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(reduce=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='none')
```

#### reduction
reduction 为'none'
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(reduction='none')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='none')
```

reduction 为'mean'
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(reduction='mean')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

reduction 为'sum'
```python
# Pytorch 写法
torch.nn.SmoothL1Loss(reduction='sum')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='sum')
```
