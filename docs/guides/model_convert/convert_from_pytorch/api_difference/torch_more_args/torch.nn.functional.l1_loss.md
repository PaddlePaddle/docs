## [ torch 参数更多 ]torch.nn.functional.l1_loss
### [torch.nn.functional.l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html?highlight=l1_loss#torch.nn.functional.l1_loss)
```python
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.l1_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/l1_loss_cn.html)
```python
paddle.nn.functional.l1_loss(input, label, reduction='mean', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input        | 表示输入的 Tensor。 |
| target        | label        | 标签，和 input 具有相同的维度，仅参数名不一致。                                      |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式。Paddle 无此参数，需要转写。                       |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式。Paddle 无此参数，需要转写。                  |
| reduction     | reduction    | 输出结果的计算方式。                                       |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.l1_loss(size_average=True)

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.l1_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.l1_loss(reduce=True)

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.l1_loss(reduce=False)

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.l1_loss(reduction='none')

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.l1_loss(reduction='mean')

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.l1_loss(reduction='sum')

# Paddle 写法
paddle.nn.functional.l1_loss(reduction='sum')
```
