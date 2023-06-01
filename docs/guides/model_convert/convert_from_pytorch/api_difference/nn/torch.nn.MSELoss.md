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
# Paddle 写法
torch.nn.MSELoss(size_average=True)

# Paddle 写法
paddle.nn.MSELoss(reduction='mean')
```

#### size_average
size_average 为 True
```python
# Pytorch 写法
torch.nn.MSELoss(size_average=True)

# Paddle 写法
paddle.nn.MSELoss(reduction='mean')

```

size_average 为 False
```python
# Pytorch 写法
torch.nn.MSELoss(size_average=False)

# Paddle 写法
paddle.nn.MSELoss(reduction='sum')
```

#### reduce
reduce 为 True
```python
# Pytorch 写法
torch.nn.MSELoss(reduce=True)

# Paddle 写法
paddle.nn.MSELoss(reduction='mean')
```

reduce 为 False
```python
# Pytorch 写法
torch.nn.MSELoss(reduce=False)

# Paddle 写法
paddle.nn.MSELoss(reduction='none')
```

#### reduction
reduction 为'none'
```python
# Pytorch 写法
torch.nn.MSELoss(reduction='none')

# Paddle 写法
paddle.nn.MSELoss(reduction='none')
```

reduction 为'mean'
```python
# Pytorch 写法
torch.nn.MSELoss(reduction='mean')

# Paddle 写法
paddle.nn.MSELoss(reduction='mean')
```

reduction 为'sum'
```python
# Pytorch 写法
torch.nn.MSELoss(reduction='sum')

# Paddle 写法
paddle.nn.MSELoss(reduction='sum')
```
