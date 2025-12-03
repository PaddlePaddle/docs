## [ torch 参数更多 ]torch.nn.functional.kl_div
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
                reduction='mean',
                log_target=False)
```

其中 PyTorch 相比 Paddle 支持更多的参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                   |
| ------------ | ------------ | ------------------------------------------------------ |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。           |
| reduction    | reduction    | 表示对输出结果的计算方式。                             |
| log_target   | log_target   | 指定目标是否属于 log 空间。                            |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.kl_div(size_average=True)

# Paddle 写法
paddle.nn.functional.kl_div(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.kl_div(size_average=False)

# Paddle 写法
paddle.nn.functional.kl_div(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.kl_div(reduce=True)

# Paddle 写法
paddle.nn.functional.kl_div(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.kl_div(reduce=False)

# Paddle 写法
paddle.nn.functional.kl_div(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.kl_div(reduction='none')

# Paddle 写法
paddle.nn.functional.kl_div(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.kl_div(reduction='mean')

# Paddle 写法
paddle.nn.functional.kl_div(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.kl_div(reduction='sum')

# Paddle 写法
paddle.nn.functional.kl_div(reduction='sum')
```
