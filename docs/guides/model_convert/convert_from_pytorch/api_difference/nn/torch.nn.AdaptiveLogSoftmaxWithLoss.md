## [ torch 参数更多 ]torch.nn.AdaptiveLogSoftmaxWithLoss
### [torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#adaptivelogsoftmaxwithloss)

```python
torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, device=None, dtype=None)
```

### [paddle.nn.AdaptiveLogSoftmaxWithLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/AdaptiveLogSoftmaxWithLoss_cn.html#adaptivelogsoftmaxwithloss)

```python
paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                         |
| ------------- | ------------ | ---------------------------------------------------------------------------- |
| in_features   | in_features  | 输入 Tensor 的特征数量。                                                     |
| n_classes     | n_classes    | 数据集中类型的个数。                                                         |
| cutoffs       | cutoffs      | 用于将 label 分配到不同存储组的截断值。                                      |
| div_value     | div_value    | 用于计算组大小的指数值。                                                     |
| head_bias     | head_bias    | 如果为 `True`，AdaptiveLogSoftmaxWithLoss 的 `head` 添加偏置项。             |
| device        | -            | 创建参数的设备，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
| dtype         | -            | 创建参数的数据类型，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
