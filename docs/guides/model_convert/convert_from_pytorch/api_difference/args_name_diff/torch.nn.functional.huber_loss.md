## [ 仅参数名不一致 ]torch.nn.functional.huber_loss
### [torch.nn.functional.huber\_loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html#torch.nn.functional.huber_loss)
```python
torch.nn.functional.huber_loss(input, target, reduction='mean', delta=1.0)
```

### [paddle.nn.functional.smooth\_l1\_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/smooth_l1_loss_cn.html#paddle.nn.functional.smooth_l1_loss)
```python
paddle.nn.functional.smooth_l1_loss(input, label, reduction='mean', delta=1.0, name=None)
```

功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                      |
| --------- | ------------ | ----------------------------------------- |
| input     | input        | 输入 Tensor。                             |
| target    | label        | 输入 input 对应的标签值，仅参数名不一致。 |
| reduction | reduction    | 指定应用于输出结果的计算方式。            |
| delta     | delta        | 损失的阈值参数。                          |
