## [ torch 参数更多 ]torch.nn.functional.binary_cross_entropy
### [torch.nn.functional.binary\_cross\_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy)
```python
torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
```

### [paddle.nn.functional.binary\_cross\_entropy](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/binary_cross_entropy_cn.html#paddle.nn.functional.binary_cross_entropy)
```python
paddle.nn.functional.binary_cross_entropy(input, label, weight=None, reduction='mean', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor。 |
| target       | label        | 标签 Tensor。 |
| weight       | weight       | 手动指定每个 batch 二值交叉熵的权重。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例


#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(size_average=True)

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(size_average=False)

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(reduce=True)

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(reduce=False)

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(reduction='none')

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(reduction='mean')

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.binary_cross_entropy(reduction='sum')

# Paddle 写法
paddle.nn.functional.binary\_cross\_entropy(reduction='sum')
```
