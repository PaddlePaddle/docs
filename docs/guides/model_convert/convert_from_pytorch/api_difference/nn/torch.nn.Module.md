## [参数不一致]torch.nn.Module

### [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module)

```python
torch.nn.Module(*args, **kwargs)
```

### [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html)

```python
paddle.nn.Layer(name_scope=None, dtype='float32')
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| :-----: | :----------: | :----------------------------------------------------------: |
|    -    |  name_scope  | PyTorch 无此参数。为 Layer 内部参数命名而采用的名称前缀。如果前缀为“mylayer”，在一个类名为 MyLayer 的 Layer 中，参数名为“mylayer_0.w_n”，其中 w 是参数的名称，n 为自动生成的具有唯一性的后缀。如果为 None，前缀名将为小写的类名。默认值为 None。 |
|    -    |    dtype     | PyTorch 无此参数。Layer 中参数数据类型。如果设置为 str，则可以是“bool”，“float16”，“float32”，“float64”，“int8”，“int16”，“int32”，“int64”，“uint8”或“uint16”。默认值为 "float32"。 |


### 转写示例
```python
# PyTorch 写法
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


# Paddle 写法
import paddle
class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)
    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp
x = paddle.randn([10, 1], 'float32')
mylayer = MyLayer()
mylayer.eval()  # set mylayer._dropout to eval mode
out = mylayer(x)
mylayer.train()  # set mylayer._dropout to train mode
out = mylayer(x)
```

