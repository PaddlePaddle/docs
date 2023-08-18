## [torch 参数更多]torch.nn.functional.binary_cross_entropy_with_logits

### [torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html?highlight=binary_cross_entropy_with_logits#torch.nn.functional.binary_cross_entropy_with_logits)

```python
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

### [paddle.nn.functional.binary_cross_entropy_with_logits](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/binary_cross_entropy_with_logits_cn.html)

```python
paddle.nn.functional.binary_cross_entropy_with_logits(logit, label, weight=None, reduction='mean', pos_weight=None, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | logit        | 表示输入的 Tensor。                                       |
| target        | label        | 标签，和 input 具有相同的维度，仅参数名不一致。                                                   |
| weight        | weight       | 类别权重。                                                |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式。 Paddle 无此参数，需要转写。                      |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式。 Paddle 无此参数，需要转写。                |
| reduction     | reduction    | 输出结果的计算方式。                                       |
| pos_weight    | pos_weight   | 正类的权重。                                              |

### 转写示例
#### size_average：是否对损失进行平均或求和
```python
# Pytorch 写法 (size_average 为‘True’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, size_average=True)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (size_average 为‘False’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, size_average=False)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='sum')
```

#### reduce：是否对损失进行平均或求和
```python
# Pytorch 写法 (reduce 为‘True’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduce=True)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (reduce 为‘False’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduce=False)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='sum')
```

#### reduction：输出结果的计算方式
```python
# Pytorch 写法 (reduction 为‘none’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='none')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='none')

# Pytorch 写法 (reduction 为‘mean’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='mean')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (reduction 为‘sum’时)
torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='sum')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(logit=a, label=target,
    reduction='sum')
```
