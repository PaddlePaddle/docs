## [ 仅 API 调用方式不一致 ]torch.utils.data.random_split

### [torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)

```python
torch.utils.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)
```

### [paddle.io.random_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/random_split_cn.html#paddle/io/random_split_cn#cn-api-paddle-io-random_split)

```python
paddle.io.random_split(dataset, lengths, generator=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

data = Data()
datasets = torch.utils.data.random_split(data, [3, 7])

# Paddle 写法
class Data(paddle.io.Dataset):
    def __init__(self):
        self.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

data = Data()
datasets = paddle.io.random_split(data, [3, 7])
```
