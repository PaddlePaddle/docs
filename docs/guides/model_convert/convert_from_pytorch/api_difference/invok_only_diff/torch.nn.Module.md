## [ 仅 API 调用方式不一致 ]torch.nn.Module

### [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

```python
torch.nn.Module(*args, **kwargs)
```

### [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#paddle/nn/Layer_cn#cn-api-paddle-nn-Layer)

```python
paddle.nn.Layer(name_scope=None, dtype='float32')
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc1.bias.fill_(0.1)
            self.fc2.weight.fill_(1.0)
            self.fc2.bias.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
inputs = torch.ones([64, 784])
result = model(inputs)

# Paddle 写法
class MLP(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=hidden_size, out_features=output_size)
        with paddle.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc1.bias.fill_(0.1)
            self.fc2.weight.fill_(1.0)
            self.fc2.bias.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
inputs = paddle.ones([64, 784])
result = model(inputs)
```
