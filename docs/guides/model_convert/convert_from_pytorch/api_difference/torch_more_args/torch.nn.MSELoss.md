## [ torch 参数更多 ]torch.nn.MSELoss
### [torch.nn.MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
```python
torch.nn.MSELoss(size_average=None,
                 reduce=None,
                 reduction='mean')
```

### [paddle.nn.MSELoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MSELoss_cn.html#paddle.nn.MSELoss)
```python
paddle.nn.MSELoss(reduction='mean')
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | 已弃用。需要转写。  |
| reduce        | -            | 已弃用。需要转写。  |
| reduction        | reduction            | 表示对输出结果的计算方式。  |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.MSELoss(size_average=False, reduce=True)
torch.nn.MSELoss(size_average=False)

# Paddle 写法
paddle.nn.MSELoss(reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.MSELoss(size_average=True, reduce=True)
torch.nn.MSELoss(reduce=True)
torch.nn.MSELoss(size_average=True)
torch.nn.MSELoss()

# Paddle 写法
paddle.nn.MSELoss(reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.MSELoss(size_average=True, reduce=False)
torch.nn.MSELoss(size_average=False, reduce=False)
torch.nn.MSELoss(reduce=False)

# Paddle 写法
paddle.nn.MSELoss(reduction='none')
```
