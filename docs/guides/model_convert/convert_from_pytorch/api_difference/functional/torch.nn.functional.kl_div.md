## [torch 参数更多]torch.nn.functional.kl_div

### [torch.nn.functional.kl_div](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html?highlight=kl_div#torch.nn.functional.kl_div)

```python
torch.nn.functional.kl_div(input,
               target,
               size_average=None,
               reduce=None,
               reduction='mean',
               log_target=False)
```

### [paddle.nn.functional.kl_div](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/kl_div_cn.html)

```python
paddle.nn.functional.kl_div(input,
                label,
                reduction='mean')
```

其中 PyTorch 相比 Paddle 支持更多的参数，具体如下：

| PyTorch      | PaddlePaddle | 备注                                                   |
| ------------ | ------------ | ------------------------------------------------------ |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduction    | reduction    | 表示对输出结果的计算方式。                             |
| log_target   | -            | 指定目标是否为 log 空间，Paddle 无此参数，暂无转写方式。 |

### 转写示例

#### size_average：是否对损失进行平均或求和
```python
# PyTorch 写法 (size_average 为‘True’时)
torch.nn.functional.kl_div(a, target, size_average=True)

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='mean')

# PyTorch 写法 (size_average 为‘False’时)
torch.nn.functional.kl_div(a, target, size_average=False)

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='sum')
```

#### reduce：是否对损失进行平均或求和
```python
# PyTorch 写法 (reduce 为‘True’时)
torch.nn.functional.kl_div(a, target, reduce=True)

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='mean')

# PyTorch 写法 (reduce 为‘False’时)
torch.nn.functional.kl_div(a, target, reduce=False)

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='sum')
```

#### reduction：输出结果的计算方式
```python
# PyTorch 写法 (reduction 为‘none’时)
torch.nn.functional.kl_div(a, target, reduction='none')

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='none')

# PyTorch 写法 (reduction 为‘mean’时)
torch.nn.functional.kl_div(a, target, reduction='mean')

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='mean')

# PyTorch 写法 (reduction 为‘sum’时)
torch.nn.functional.kl_div(a, target, reduction='sum')

# Paddle 写法
paddle.nn.functional.kl_div(logit=a, label=target,
    reduction='sum')
```
