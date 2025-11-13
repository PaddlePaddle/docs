## [ torch 参数更多 ]torch.nn.L1Loss
### [torch.nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html?highlight=l1loss#torch.nn.L1Loss)
```python
torch.nn.L1Loss(size_average=None,
                reduce=None,
                reduction='mean')
```

### [paddle.nn.L1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/L1Loss_cn.html#l1loss)
```python
paddle.nn.L1Loss(reduction='mean',
                 name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | 已弃用。需要转写。  |
| reduce        | -            | 已弃用。需要转写。  |
| reduction        | reduction            | 表示对输出结果的计算方式。  |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.L1Loss(size_average=True)

# Paddle 写法
paddle.nn.L1Loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.L1Loss(size_average=False)

# Paddle 写法
paddle.nn.L1Loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.L1Loss(reduce=True)

# Paddle 写法
paddle.nn.L1Loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.L1Loss(reduce=False)

# Paddle 写法
paddle.nn.L1Loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.L1Loss(reduction='none')

# Paddle 写法
paddle.nn.L1Loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.L1Loss(reduction='mean')

# Paddle 写法
paddle.nn.L1Loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.L1Loss(reduction='sum')

# Paddle 写法
paddle.nn.L1Loss(reduction='sum')
```
