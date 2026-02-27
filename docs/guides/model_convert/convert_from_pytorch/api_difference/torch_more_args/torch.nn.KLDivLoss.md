## [ torch 参数更多 ]torch.nn.KLDivLoss
### [torch.nn.KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss)
```python
torch.nn.KLDivLoss(size_average=None,
                   reduce=None,
                   reduction='mean',
                   log_target=False)
```

### [paddle.nn.KLDivLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/KLDivLoss_cn.html#paddle.nn.KLDivLoss)
```python
paddle.nn.KLDivLoss(reduction='mean',
                    log_target=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。            |
| reduce        | -            | PyTorch 已弃用， Paddle 无此参数，需要转写。 |
| reduction        | reduction            | 表示对输出结果的计算方式。  |
| log_target    | log_target   | 指定目标是否属于 log 空间。                            |

### 转写示例
#### size_average/reduce：对应到 reduction 为 sum
```python
# PyTorch 写法
torch.nn.KLDivLoss(size_average=False, reduce=True)
torch.nn.KLDivLoss(size_average=False)

# Paddle 写法
paddle.nn.KLDivLoss(reduction='sum')
```

#### size_average/reduce：对应到 reduction 为 mean
```python
# PyTorch 写法
torch.nn.KLDivLoss(size_average=True, reduce=True)
torch.nn.KLDivLoss(reduce=True)
torch.nn.KLDivLoss(size_average=True)
torch.nn.KLDivLoss()

# Paddle 写法
paddle.nn.KLDivLoss(reduction='mean')
```

#### size_average/reduce：对应到 reduction 为 none
```python
# PyTorch 写法
torch.nn.KLDivLoss(size_average=True, reduce=False)
torch.nn.KLDivLoss(size_average=False, reduce=False)
torch.nn.KLDivLoss(reduce=False)

# Paddle 写法
paddle.nn.KLDivLoss(reduction='none')
```
