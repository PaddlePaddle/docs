## [ 仅 paddle 参数更多 ]torch.utils.data.BatchSampler
### [torch.utils.data.BatchSampler](https://pytorch.org/docs/stable/data.html?highlight=batchsampler#torch.utils.data.BatchSampler)

```python
torch.utils.data.BatchSampler(sampler,
                              batch_size,
                              drop_last)
```

### [paddle.io.BatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html#batchsampler)

```python
paddle.io.BatchSampler(dataset=None,
                       sampler=None,
                       shuffle=Fasle,
                       batch_size=1,
                       drop_last=False)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注    |
| --------- | -------------| ---------- |
| sampler       | sampler      | 底层取样器，PyTorch 可为 Sampler 或 Iterable 数据类型，Paddle 可为 Sampler 数据类型。 |
| -             | dataset      | 此参数必须是 paddle.io.Dataset 或 paddle.io.IterableDataset 的一个子类实例或实现了 __len__ 的 Python 对象，用于生成样本下标，PyTorch 无此参数，Paddle 保持默认即可。              |
| -             | shuffle      | 是否需要在生成样本下标时打乱顺序，PyTorch 无此参数，Paddle 保持默认即可。 |


### 转写示例
#### sampler(Iterable)：底层取样器
```python
# 若 sampler 为 Iterable 数据类型，则需要按如下方式转写
# Pytorch 写法
torch.utils.data.BatchSampler(sampler=[1., 2., 3., 4.], batch_size=3, drop_last = False)

# Paddle 写法
sampler = [1.0, 2.0, 3.0, 4.0]
sampler = sampler if issubclass(sampler.__class__, paddle.fluid.dataloader.
    sampler.Sampler().__class__) else paddle.io.Sampler(sampler)
paddle.io.BatchSampler(sampler=sampler, batch_size=3, drop_last=False)
```
