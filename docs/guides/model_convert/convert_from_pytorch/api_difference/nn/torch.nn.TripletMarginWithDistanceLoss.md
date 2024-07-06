## [ 参数完全一致 ] torch.nn.TripletMarginWithDistanceLoss

### [torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html)

```python
torch.nn.TripletMarginWithDistanceLoss(*, distance_function=None, margin=1.0, swap=False, reduction='mean')
```

### [paddle.nn.TripletMarginWithDistanceLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/TripletMarginWithDistanceLoss_cn.html#tripletmarginwithdistanceloss)

```python
paddle.nn.TripletMarginWithDistanceLoss(distance_function=None, margin: float = 1.0, swap: bool = False, reduction: str = 'mean', name: str = None)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle | 备注                                           |
| ----------------- | ------------ | ---------------------------------------------- |
| distance_function | distance_function | 手动指定范数，参数完全一致。                                 |
| margin            | margin            | 手动指定间距，参数完全一致。                                 |
| swap              | swap         | 默认为 False。                                 |
| reduction         | reduction           | 指定应用于输出结果的计算方式，参数完全一致。 |
