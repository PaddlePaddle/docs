## [ 仅 API 调用方式不一致 ]torch.utils.data.Sampler

### [torch.utils.data.Sampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)

```python
torch.utils.data.Sampler(data_source)
```

### [paddle.io.Sampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/Sampler_cn.html#paddle.io.Sampler)

```python
paddle.io.Sampler(data_source)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class MySampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

# Paddle 写法
class MySampler(paddle.io.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
```
