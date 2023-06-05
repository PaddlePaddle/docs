## [ torch 参数更多 ]torch.nn.functional.cross_entropy

### [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html?highlight=cross_#torch.nn.functional.cross_entropy)

```python
torch.nn.functional.cross_entropy(input,
                                 target,
                                 weight=None,
                                 size_average=None,
                                 ignore_index=- 100,
                                 reduce=None,
                                 reduction='mean',
                                 label_smoothing=0.0)
```

### [paddle.nn.functional.cross_entropy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/cross_entropy_cn.html)

```python
paddle.nn.functional.cross_entropy(input,
                                   label,
                                   weight=None,
                                   ignore_index=- 100,
                                   reduction='mean',
                                   soft_label=False,
                                   axis=- 1,
                                   name=None)
```
两者功能一致，torch 参数更多，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 表示预测的 Tensor 。                                     |
| target          | label         | 表示真实的 Tensor 。                                     |
| weight          | weight         | 表示权重。                                     |
| size_average    | -         | 已弃用 。                                     |
| ignore_index          | ignore_index         | 表示忽略的标签值 。                                     |
| reduce          | -         | 已弃用 。                                     |
| reduction          | reduction         | 表示应用于输出结果的计算方式 。                                     |
| label_smoothing |    - | 指定计算损失时的平滑量， Paddle 无此功能，暂无转写方式。|
| -               | soft_label | 指明 label 是否为软标签， Pytorch 无此参数， Paddle 保持默认即可。|
| -                  | axis | 进行 softmax 计算的维度索引， Pytorch 无此参数， Paddle 保持默认即可。|

### 转写示例
#### size_average
size_average 为 True
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,size_average=True)

# Paddle 写法
paddle.nn.functional.cross_entropy(x,y,reduction='mean')

```

size_average 为 False
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,size_average=False)

# Paddle 写法
paddle.nn.functional.cross_entropy(x,y,reduction='sum')
```

#### reduce
reduce 为 True
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,reduce=True)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(reduction='mean')
```

reduce 为 False
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,reduce=False)

# Paddle 写法
paddle.nn.BCEWithLogitsLoss(reduction='none')
```

#### reduction
reduction 为'none'
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,reduction='none')

# Paddle 写法
paddle.nn.functional.cross_entropy(x,y,reduction='none')
```

reduction 为'mean'
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,reduction='mean')

# Paddle 写法
paddle.nn.functional.cross_entropy(x,y,reduction='mean')
```

reduction 为'sum'
```python
# Pytorch 写法
torch.nn.functional.cross_entropy(x,y,reduction='sum')

# Paddle 写法
paddle.nn.functional.cross_entropy(x,y,reduction='sum')
```
