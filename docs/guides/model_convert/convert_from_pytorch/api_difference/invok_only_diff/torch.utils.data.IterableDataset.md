## [ 仅 API 调用方式不一致 ]torch.utils.data.IterableDataset

### [torch.utils.data.IterableDataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

```python
torch.utils.data.IterableDataset(*args, **kwargs)
```

### [paddle.io.IterableDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/IterableDataset_cn.html#paddle.io.IterableDataset)

```python
paddle.io.IterableDataset()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

# Paddle 写法
class MyIterableDataset(paddle.io.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))
```
