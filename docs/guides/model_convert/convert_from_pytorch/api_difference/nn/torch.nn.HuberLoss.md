## [参数完全一致]torch.nn.HuberLoss

### [torch.nn.HuberLoss](https://pytorch.org/docs/1.13/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss)

```python
torch.nn.HuberLoss(reduction='mean', delta=1.0)
```

### [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SmoothL1Loss_cn.html)

```python
paddle.nn.SmoothL1Loss(reduction='mean', delta=1.0, name=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                           |
| --------- | ------------ | ------------------------------ |
| reduction | reduction    | 指定应用于输出结果的计算方式。 |
| delta     | delta        | 损失的阈值参数。               |
