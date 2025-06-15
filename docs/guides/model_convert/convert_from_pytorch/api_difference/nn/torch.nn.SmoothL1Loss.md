## [torch 参数更多 ]torch.nn.SmoothL1Loss
### [torch.nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=smoothl1loss#torch.nn.SmoothL1Loss)

```python
torch.nn.SmoothL1Loss(size_average=None,
                      reduce=None,
                      reduction='mean',
                      beta=1.0)
```

### [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SmoothL1Loss_cn.html#smoothl1loss)

```python
paddle.nn.SmoothL1Loss(reduction='mean',
                       delta=1.0,
                       is_huber=True,
                       name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示应用于输出结果的计算方式。 |
| beta          | delta        | SmoothL1Loss 损失的阈值参数，beta 不为 1.0 时 Paddle 不支持，暂无转写方式。  |
| -          | is_huber         | 参数为 True 时，函数为 Huber 损失。参数为 False 时，函数为 Huber 损失除以 delta，此时 paddle 和 pytorch 一致                     |

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
# PyTorch 写法
torch.nn.SmoothL1Loss(size_average=True)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(size_average=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='sum')
```

#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(reduce=True)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

reduce 为 False
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(reduce=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='none')
```

#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(reduction='none')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='none')
```

reduction 为'mean'
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(reduction='mean')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean')
```

reduction 为'sum'
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(reduction='sum')

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='sum')
```

#### beta
```python
# PyTorch 的 beta 参数转化为 delta 参数
a=0.8

# PyTorch 写法
torch.nn.SmoothL1Loss(beta=a)

# Paddle 写法
paddle.nn.SmoothL1Loss(delta=a, is_huber=False)
```
