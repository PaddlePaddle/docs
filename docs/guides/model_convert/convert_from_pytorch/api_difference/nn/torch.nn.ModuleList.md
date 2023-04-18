## [仅参数名不一致]torch.nn.ModuleList

### [torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html?highlight=torch+nn+modulelist#torch.nn.ModuleList)

```python
torch.nn.ModuleList(modules=None)
```

### [paddle.nn.LayerList](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerList_cn.html)

```python
paddle.nn.LayerList(sublayers=None)
```

两者功能一致，参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| :-----: | :----------: | :----------------------------------------------------------: |
| modules |  sublayers   | 键值对的可迭代对象，值的类型为 paddle.nn.Layer，仅参数名不一致。 |


### 转写示例
```python
# PyTorch 写法
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


# Paddle 写法
import paddle
import numpy as np

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # LayerList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

