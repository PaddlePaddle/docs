## [ paddle 参数更多 ]torch.utils.data.BatchSampler
### [torch.utils.data.BatchSampler](https://pytorch.org/docs/stable/data.html?highlight=batchsampler#torch.utils.data.BatchSampler)

```python
torch.utils.data.BatchSampler(sampler,
                              batch_size,
                              drop_last)
```

### [paddle.io.BatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/BatchSampler_cn.html#batchsampler)

```python
paddle.io.BatchSampler(dataset=None,
                       sampler=None,
                       shuffle=Fasle,
                       batch_size=1,
                       drop_last=False)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注    |
| --------- | -------------| ---------- |
| sampler       | sampler      | 底层取样器，可为 Sampler 或 Iterable 数据类型。 |
| batch_size    | batch_size   | 每 mini-batch 中包含的样本数。PyTorch 无默认值，Paddle 默认值为 1。           |
| drop_last     | drop_last    | 是否需要丢弃最后无法凑整一个 mini-batch 的样本。PyTorch 无默认值，Paddle 默认值为 False。      |
| -             | dataset      | 此参数必须是 paddle.io.Dataset 或 paddle.io.IterableDataset 的一个子类实例或实现了 __len__ 的 Python 对象，用于生成样本下标，PyTorch 无此参数，Paddle 保持默认即可。              |
| -             | shuffle      | 是否需要在生成样本下标时打乱顺序，PyTorch 无此参数，Paddle 保持默认即可。 |
