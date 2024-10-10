## [torch 参数更多]torchvision.transforms.functional.normalize

### [torchvision.transforms.functional.normalize](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.normalize.html)

```python
torchvision.transforms.functional.normalize(img, mean, std, inplace = False)
```

### [paddle.vision.transforms.normalize](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/normalize_cn.html)

```python
paddle.vision.transforms.normalize(img, mean = 0.0, std = 1.0, data_format = 'CHW', to_rgb = False, keys = None)
```

两者功能一致，但 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| img  | img  | 用于归一化的数据。 |
| mean                   | mean  | 用于每个通道归一化的均值。                                   |
| std                    | std   | 用于每个通道归一化的标准差值。                               |
| inplace          | -                                     | 是否原地修改，Paddle 无此参数，需要转写。                               |
| -                                | data_format                      | 用于指定数据格式，默认为 'CHW'，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                                | to_rgb                          | 是否将图像转换为 RGB 格式，默认为 False，PyTorch 无此参数，Paddle 保持默认即可。 |
| -                                | keys         | Paddle 支持 `keys` 参数，默认为 None，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### 基本归一化

```python
# PyTorch 写法
import torch
import torchvision.transforms.functional as F
mean = 0.5, 0.5, 0.5
std = [0.5, 0.5, 0.5]
img = torch.tensor([
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.5, 0.5], [0.5, 0.5]]
])
result = F.normalize(img, mean=mean, std=std)

# Paddle 写法
import paddle
mean = 0.5, 0.5, 0.5
std = [0.5, 0.5, 0.5]
img = paddle.to_tensor(data=[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5,
    0.5]], [[0.5, 0.5], [0.5, 0.5]]])
result = paddle.vision.transforms.normalize(img=img, mean=mean, std=std)

```
