## [仅参数名不一致]torch.nn.ModuleDict

### [torch.nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html?highlight=torch+nn+moduledict#torch.nn.ModuleDict)

```python
torch.nn.ModuleDict(modules=None)
```

### [paddle.nn.LayerDict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerDict_cn.html)

```python
paddle.nn.LayerDict(sublayers=None)
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
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


# Paddle 写法
import paddle
import numpy as np
from collections import OrderedDict

sublayers = OrderedDict([
    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
])

layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

l = layers_dict['conv1d']

for k in layers_dict:
    l = layers_dict[k]

len(layers_dict)
#3

del layers_dict['conv2d']
len(layers_dict)
#2

conv1d = layers_dict.pop('conv1d')
len(layers_dict)
#1

layers_dict.clear()
len(layers_dict)
#0
```

