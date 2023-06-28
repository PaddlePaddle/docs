## [参数完全一致] torch.utils.data.WeightedRandomSampler

### [torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

```
torch.utils.data.WeightedRandomSampler(weights,
                       num_samples,
                       replacement=True,
                       generator=None)
```

### [paddle.io.WeightedRandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/WeightedRandomSampler_cn.html#paddle.io.WeightedRandomSampler)

```
paddle.io.WeightedRandomSampler(weights,
                num_samples,
                replacement=True)
```

两者参数完全一致，具体如下：

### 参数差异

| PyTorch     | PaddlePaddle | 备注                                                                 |
| ----------- | ------------ | -------------------------------------------------------------------- |
| weights     | weights      | 权重序列，需要时 numpy 数组， paddle.Tensor， list 或者 tuple 类型。 |
| num_samples | num_samples  | 采样样本数。                                                         |
| replacement | replacement  | 是否采用有放回的采样。                                               |
| generator   | -            | 采样中的生成器，Paddle 无此参数，可忽略。                            |
