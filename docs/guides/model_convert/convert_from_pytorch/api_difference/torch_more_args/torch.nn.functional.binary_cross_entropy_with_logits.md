## [ torch 参数更多 ]torch.nn.functional.binary_cross_entropy_with_logits
### [torch.nn.functional.binary\_cross\_entropy\_with\_logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html#torch.nn.functional.binary_cross_entropy_with_logits)
```python
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

### [paddle.nn.functional.binary\_cross\_entropy\_with\_logits](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/binary_cross_entropy_with_logits_cn.html#paddle.nn.functional.binary_cross_entropy_with_logits)
```python
paddle.nn.functional.binary_cross_entropy_with_logits(logit, label, weight=None, reduction='mean', pos_weight=None, name=None)
```

其中 `input` 是 Paddle 参数 `logit` 的别名，`target` 是 Paddle 参数 `label` 的别名。

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | logit        | 表示输入的 Tensor。Paddle 兼容 `input` 别名，无需改写。                                       |
| target        | label        | 标签，和 input 具有相同的维度。Paddle 兼容 `target` 别名，无需改写。                                                   |
| weight        | weight       | 类别权重。                                                |
| size_average  | -            | 已废弃，和 reduce 组合决定损失计算方式。 Paddle 无此参数，需要转写。                      |
| reduce        | -            | 已废弃，和 size_average 组合决定损失计算方式。 Paddle 无此参数，需要转写。                |
| reduction     | reduction    | 输出结果的计算方式。                                       |
| pos_weight    | pos_weight   | 正类的权重。                                              |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(size_average=True)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(size_average=False)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(reduce=True)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(reduce=False)

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(reduction='none')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(reduction='mean')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy_with_logits(reduction='sum')

# Paddle 写法
paddle.nn.functional.binary_cross_entropy_with_logits(reduction='sum')
```
