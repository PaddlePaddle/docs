## [ torch 参数更多 ]torch.nn.SmoothL1Loss
### [torch.nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=smoothl1loss#torch.nn.SmoothL1Loss)
```python
torch.nn.SmoothL1Loss(size_average=None,
                      reduce=None,
                      reduction='mean',
                      beta=1.0)
```

### [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SmoothL1Loss_cn.html#smoothl1loss)
```python
paddle.nn.SmoothL1Loss(reduction='mean',
                       delta=1.0,
                       is_huber=True,
                       name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示应用于输出结果的计算方式。 |
| beta          | delta        | SmoothL1Loss 损失的阈值参数，beta 不为 1.0 时 Paddle 不支持，暂无转写方式。  |
| -          | is_huber         | 控制 huber_loss 与 smooth_l1_loss 的开关，Paddle 需设置为 False 。                    |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(beta=1.0, size_average=False, reduce=True)
torch.nn.SmoothL1Loss(beta=1.0, size_average=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='sum', is_huber=False)
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(size_average=True, reduce=True)
torch.nn.SmoothL1Loss(reduce=True)
torch.nn.SmoothL1Loss(size_average=True)
torch.nn.SmoothL1Loss()

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='mean',is_huber=False)
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.SmoothL1Loss(size_average=True, reduce=False)
torch.nn.SmoothL1Loss(size_average=False, reduce=False)
torch.nn.SmoothL1Loss(reduce=False)

# Paddle 写法
paddle.nn.SmoothL1Loss(reduction='none',is_huber=False)
```
