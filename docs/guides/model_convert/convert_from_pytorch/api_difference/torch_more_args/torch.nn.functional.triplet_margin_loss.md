## [ torch 参数更多 ]torch.nn.functional.triplet_margin_loss
### [torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.triplet_margin_loss.html?highlight=triplet_margin_loss#torch.nn.functional.triplet_margin_loss)
```python
torch.nn.functional.triplet_margin_loss(anchor,
                positive,
                negative,
                margin=1.0,
                p=2,
                eps=1e-06,
                swap=False,
                size_average=None,
                reduce=None,
                reduction='mean')
```

### [paddle.nn.functional.triplet_margin_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/triplet_margin_loss_cn.html)
```python
paddle.nn.functional.triplet_margin_loss(input,
                positive,
                negative,
                margin: float = 1.0,
                p: float = 2.0,
                epsilon: float = 1e-6,
                swap: bool = False,
                reduction: str = 'mean',
                name: str = None)
```

其中 PyTorch 相⽐ Paddle ⽀持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| anchor          | input         | 输入 Tensor，仅参数名不一致。                        |
| positive          | positive         | 输入正样本。                                 |
| negative          | negative         | 输入负样本。                                     |
| margin          | margin         |  手动指定间距。                                  |
| p          | p         | 指定范数。                                 |
| eps          | epsilon         | 防止除数为零的常数。                                  |
| swap          | swap         | 是否进行交换。                                  |
| size_average          | -         | 已弃用。 需要转写。                                     |
| reduce          | -         | 已弃用。需要转写。                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'。             |

### 转写示例

#### size_average
size_average 为 True
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(size_average=True)

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='mean')
```

size_average 为 False
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='sum')
```
#### reduce
reduce 为 True
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(size_average=False)

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='sum')
```
reduce 为 False
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(reduce=False)

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='none')
```
#### reduction
reduction 为'none'
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(reduction='none')

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='none')
```
reduction 为'mean'
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(reduction='mean')

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='mean')
```
reduction 为'sum'
```python
# PyTorch 写法
torch.nn.functional.triplet_margin_loss(reduction='sum')

# Paddle 写法
paddle.nn.functional.triplet_margin_loss(reduction='sum')
```
