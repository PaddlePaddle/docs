## [ 仅 API 调用方式不一致 ]torch.utils.data.Dataset

### [torch.utils.data.Dataset](https://pytorch.org/docs/stable/generated/torch.utils.data.Dataset.html#torch.utils.data.Dataset)

```python
torch.utils.data.Dataset(*args, **kwargs)
```

### [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/Dataset_cn.html#paddle/io/Dataset_cn#cn-api-paddle-io-Dataset)

```python
paddle.io.Dataset()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.x = [1, 2, 3, 4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

# Paddle 写法
class Data(paddle.io.Dataset):
    def __init__(self):
        self.x = [1, 2, 3, 4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)
```
