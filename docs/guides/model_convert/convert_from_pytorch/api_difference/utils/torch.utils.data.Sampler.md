## [参数完全一致]torch.utils.data.Sampler

### [torch.utils.data.Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)

```python
torch.utils.data.Sampler(data_source)
```

### [paddle.io.Sampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Sampler_cn.html)

```python
paddle.io.Sampler(data_source)
```

paddle 参数和 torch 参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                   |
| ----------- | ------------ | -------------------------------------- |
| data_source | data_source  | Dataset 或者 IterableDataset 的子类实现。 |
