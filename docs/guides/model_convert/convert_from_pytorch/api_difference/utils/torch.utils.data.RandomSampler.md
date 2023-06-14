## [参数完全一致]torch.utils.data.RandomSampler

### [torch.utils.data.RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler)

```python
torch.utils.data.RandomSampler(data_source,
                   replacement=False,
                   num_samples=None,
                   generator=None)
```

### [paddle.io.RandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/RandomSampler_cn.html)

```python
paddle.io.RandomSampler(data_source=None,
            replacement=False,
            num_samples=None,
            generator=None)
```

当 replacement 为 False 时，paddle 不允许指定 num_samples，只能按整个数据集采样，pytorch 可以指定采样 num_samples 个数据集；当 replacement 为 True 时，两者完全一致（注：Paddle API 可能后续调整）。Paddle 相比 Pytorch 的映射关系如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                                                         |
| ----------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| data_source | data_source  | 此参数必须是 paddle.io.Dataset 或 paddle.io.IterableDataset 的一个子类实例或实现了__len__ 的 Python 对象，用于生成样本下标。 |
| replacement | replacement  | 是否放回采样。如果是 True，意味着同一个元素可以被采样多次。默认为 False 。                                                    |
| num_samples | num_samples  | 按照此参数采集对应的样本数。默认为 None。                                                                                     |
| generator   | generator    | 指定采样 data_source 的采样器，默认值为 None。                                                                               |
