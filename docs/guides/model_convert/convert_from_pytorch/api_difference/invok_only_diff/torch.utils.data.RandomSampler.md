## [ 仅 API 调用方式不一致 ]torch.utils.data.RandomSampler
### [torch.utils.data.RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler)
```python
torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

### [paddle.io.RandomSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/RandomSampler_cn.html#paddle.io.RandomSampler)
```python
paddle.io.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
torch.utils.data.RandomSampler(data_source=[1, 2, 3, 4, 5], replacement=False, num_samples=None, generator=None)

# Paddle 写法
paddle.io.RandomSampler(data_source=[1, 2, 3, 4, 5], replacement=False, num_samples=None, generator=None)
```
