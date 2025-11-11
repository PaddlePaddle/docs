## [ torch 参数更多 ]torch.nn.functional.poisson_nll_loss
### [torch.nn.functional.poisson\_nll\_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.poisson_nll_loss.html)
```python
torch.nn.functional.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

### [paddle.nn.functional.poisson\_nll\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/poisson_nll_loss_cn.html#poisson-nll-loss)
```python
paddle.nn.functional.poisson_nll_loss(input, label, log_input=True, full=False, epsilon=1e-8, reduction='mean', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注 |
| ------------ | ------------ | -- |
| input        | input        | 输入 Tensor。 |
| target       | label        | 标签 Tensor，仅参数名不一致。 |
| log_input    | log_input    | 输入是否为对数函数映射后结果。 |
| full         | full         | 是否在损失计算中包括 Stirling 近似项。 |
| size_average | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| eps          | epsilon      | 在 log_input 为 True 时使用的常数小量，仅参数名不一致。 |
| reduce       | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。                  |
| reduction    | reduction    | 指定应用于输出结果的计算方式。 |

### 转写示例

#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(size_average=True)

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(reduce=False)

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(reduction='none')

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(reduction='mean')

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.poisson_nll_loss(reduction='sum')

# Paddle 写法
paddle.nn.functional.poisson\_nll\_loss(reduction='sum')
```
