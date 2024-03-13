## [ 参数完全一致 ] torch.utils.data.RandomSampler

### [torch.utils.data.RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler)

```python
torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

### [paddle.io.RandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/RandomSampler_cn.html#paddle.io.RandomSampler)

```python
paddle.io.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

两者参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                 |
| ----------- | ------------ | -------------------------------------------------------------------- |
| data_source | data_source  | Dataset 或 IterableDataset 的一个子类实例或实现了 `__len__` 的 Python 对象。            |
| replacement | replacement  | 如果为 False 则会采样整个数据集。    |
| num_samples | num_samples  | 如果 replacement 设置为 True 则按此参数采集对应的样本数。    |
| generator   | generator    | 指定采样 data_source 的采样器。 |
