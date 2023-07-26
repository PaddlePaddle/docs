## [torch 参数更多]torch.nn.functional.l1_loss

### [torch.nn.functional.l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html?highlight=l1_loss#torch.nn.functional.l1_loss)

```python
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.l1_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/l1_loss_cn.html)

```python
paddle.nn.functional.l1_loss(input, label, reduction='mean', name=None)
```

两者功能一致，torch 参数多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input        | 表示输入的 Tensor。
| target        | label        | 标签，和 input 具有相同的维度，仅参数名不一致。                                      |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式。Paddle 无此参数，需要进行转写。                       |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式。Paddle 无此参数，需要进行转写。                  |
| reduction     | reduction    | 输出结果的计算方式                                       |

### 转写示例
#### size_average：是否对损失进行平均或求和
```python
# Pytorch 写法 (size_average 为‘True’时)
torch.nn.functional.l1_loss(a, target, size_average=True)

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (size_average 为‘False’时)
torch.nn.functional.l1_loss(a, target, size_average=False)

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='sum')
```

#### reduce：是否对损失进行平均或求和
```python
# Pytorch 写法 (reduce 为‘True’时)
torch.nn.functional.l1_loss(a, target, reduce=True)

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (reduce 为‘False’时)
torch.nn.functional.l1_loss(a, target, reduce=False)

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='sum')
```

#### reduction：输出结果的计算方式
```python
# Pytorch 写法 (reduction 为‘none’时)
torch.nn.functional.l1_loss(a, target, reduction='none')

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='none')

# Pytorch 写法 (reduction 为‘mean’时)
torch.nn.functional.l1_loss(a, target, reduction='mean')

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='mean')

# Pytorch 写法 (reduction 为‘sum’时)
torch.nn.functional.l1_loss(a, target, reduction='sum')

# Paddle 写法
paddle.nn.functional.l1_loss(logit=a, label=target,
    reduction='sum')
```
