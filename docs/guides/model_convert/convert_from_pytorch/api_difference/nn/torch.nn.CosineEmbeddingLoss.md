## [torch 参数更多]torch.nn.CosineEmbeddingLoss

### [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html?highlight=cosineembeddingloss#torch.nn.CosineEmbeddingLoss)

```
torch.nn.CosineEmbeddingLoss(margin=0.0,
                 size_average=None,
                 reduce=None,
                 reduction="mean")
```

### [paddle.nn.CosineEmbeddingLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/CosineEmbeddingLoss_cn.html)

```
paddle.nn.CosineEmbeddingLoss(margin=0,
                  reduction="mean")
```

其中 Pytorch 相比 Paddle 支持更多的参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                             |
| ------------ | ------------ | ---------------------------------------------------------------- |
| margin       | margin       | 可以设置的范围为[-1, 1]，建议设置的范围为[0, 0.5]。其默认为 0 。 |
| size_average | -            | PyTorch 已弃用，Paddle 无此参数，需要转写。表示忽略的一个标签    |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                     |
| reduction    | reduction    | 表示应用于输出结果的计算方式。                                   |

### 转写示例

#### size_average

size_average 为 True

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLoss(size_average=True)

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='mean')

```

size_average 为 False

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(size_average=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='sum')
```

#### reduce

reduce 为 True

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(reduce=True)

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='mean')
```

reduce 为 False

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(reduce=False)

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='none')
```

#### reduction

reduction 为'none'

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(reduction='none')

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='none')
```

reduction 为'mean'

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(reduction='mean')

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='mean')
```

reduction 为'sum'

```python
# Pytorch 写法
torch.nn.CosineEmbeddingLos(reduction='sum')

# Paddle 写法
paddle.nn.CosineEmbeddingLos(reduction='sum')
```
