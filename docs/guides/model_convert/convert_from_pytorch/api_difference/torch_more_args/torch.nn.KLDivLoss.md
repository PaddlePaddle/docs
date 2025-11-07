## [ torch 参数更多 ]torch.nn.KLDivLoss
### [torch.nn.KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss)
```python
torch.nn.KLDivLoss(size_average=None,
                   reduce=None,
                   reduction='mean',
                   log_target=False)
```

### [paddle.nn.KLDivLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/KLDivLoss_cn.html#kldivloss)
```python
paddle.nn.KLDivLoss(reduction='mean',
                    log_target=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。            |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示对输出结果的计算方式。  |
| log_target    | log_target   | 指定目标是否属于 log 空间。                            |

### 转写示例
#### size_average
size_average 为 True

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, size_average=True)

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='mean')
```

size_average 为 False

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, size_average=False)

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='sum')
```

#### reduce
reduce 为 True

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, reduce=True)

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='mean')
```

reduce 为 False

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, reduce=False)

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='none')
```

#### reduction
reduction 为'none'

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, reduction='none')

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='none')
```

reduction 为'mean'

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, reduction='mean')

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='mean')
```

reduction 为'sum'

```python
# PyTorch 写法
torch.nn.KLDivLoss(weight=w, reduction='sum')

# Paddle 写法
paddle.nn.KLDivLoss(weight=w, reduction='sum')
```
