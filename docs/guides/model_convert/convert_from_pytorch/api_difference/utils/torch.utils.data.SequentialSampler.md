## [参数完全一致]torch.utils.data.Sampler

### [torch.utils.data.SequentialSampler](https://pytorch.org/docs/stable/data.html?highlight=torch+utils+data+sequentialsampler#torch.utils.data.SequentialSampler)

```python
torch.utils.data.SequentialSampler(data_source)
```

### [paddle.io.SequenceSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/SequenceSampler_cn.html#sequencesampler)

```python
paddle.io.SequenceSampler(data_source)
```

paddle 参数和 torch 参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                   |
| ----------- | ------------ | -------------------------------------- |
| data_source | data_source  | Dataset 或者 IterableDataset 的子类实现。 |


### 转写示例
```python
# Pytorch 写法
torch.utils.data.SequentialSampler(dataset)

# Paddle 写法
paddle.io.SequenceSampler(dataset)
```
