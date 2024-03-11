## [ 参数完全一致 ]torch.utils.data.SequentialSampler

### [torch.utils.data.SequentialSampler](https://pytorch.org/docs/stable/generated/torch.utils.data.SequentialSampler.html)

```python
torch.utils.data.SequentialSampler(data_source)
```

### [paddle.io.SequenceSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/SequenceSampler_cn.html#sequencesampler)

```python
paddle.io.SequenceSampler(data_source)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注 |
| ----------- | ------------ | -- |
| data_source | data_source  | Dataset 或 IterableDataset 的一个子类实例或实现了 `__len__` 的 Python 对象。 |
