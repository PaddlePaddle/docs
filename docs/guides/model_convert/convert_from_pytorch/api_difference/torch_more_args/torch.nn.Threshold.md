## [torch 参数更多]torch.nn.Threshold

### [torch.nn.Threshold](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html#torch.nn.Threshold)

```python
torch.nn.Threshold(threshold, value, inplace=False)
```

### [paddle.nn.ThresholdedReLU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/ThresholdedReLU_cn.html)

```python
paddle.nn.ThresholdedReLU(threshold=1.0, value=0.0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                                            |
| --------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| threshold | threshold    | ThresholdedReLU 激活计算公式中的 threshold 值。                                                                 |
| value     | value        | 不在指定 threshold 范围时的值。   |
| inplace   | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
