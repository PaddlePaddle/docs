## [ torch 参数更多 ]torch.nn.functional.soft_margin_loss
### [torch.nn.functional.soft\_margin\_loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.soft_margin_loss.html#torch.nn.functional.soft_margin_loss)
```python
torch.nn.functional.soft_margin_loss(input,
                             target,
                             size_average=None,
                             reduce=None,
                             reduction='mean')
```

### [paddle.nn.functional.soft\_margin\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/soft_margin_loss_cn.html#paddle.nn.functional.soft_margin_loss)
```python
paddle.nn.functional.soft_margin_loss(input,
                              label,
                              reduction='mean',
                              name=None)
```

其中 PyTorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 输入 Tensor 。                                    |
| target          | label         | 输入 Tensor 对应的标签值，仅参数名不一致。               |
| size_average          | -         | 已弃用。需要转写。                                      |
| reduce          | -         | 已弃用。需要转写。                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'。        |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(size_average=True)

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(reduce=True)

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(reduce=False)

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(reduction='none')

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(reduction='mean')

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.soft_margin_loss(reduction='sum')

# Paddle 写法
paddle.nn.functional.soft_margin_loss(reduction='sum')
```
