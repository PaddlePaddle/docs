## [输入参数类型不一致]torchvision.transforms.functional.perspective

### [torchvision.transforms.functional.perspective](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.perspective.html#perspective)

```python
torchvision.transforms.functional.perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None
)
```

### [paddle.vision.transforms.perspective](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/perspective_cn.html#cn-api-paddle-vision-transforms-perspective)

```python
paddle.vision.transforms.perspective(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    startpoints: List[List[float]],
    endpoints: List[List[float]],
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0
) -> Union[PIL.Image.Image, np.ndarray, paddle.Tensor]

```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ---------------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| img                      | img  | 输入图片。 |
| startpoints                  | startpoints         | 数据类型不同，torchvision 使用 int，Paddle 使用 float       |
| endpoints                    | endpoints           | 数据类型不同，torchvision 使用 int，Paddle 使用 float       |
| interpolation       | interpolation              | 参数名相同但类型不同，Paddle 使用 str 替代 InterpolationMode |
| fill         | fill             | Paddle 支持更多类型，支持 int、list、tuple                    |
| -                                              | keys          | Paddle 支持 keys 参数 |
| antialias                      | -                                      | Paddle 不支持 antialias 参数                                 |


### 转写示例

```python
# PyTorch 写法
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
img = Image.open('path_to_image.jpg')
startpoints = [[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]]
endpoints = [[10, 10], [img.width - 10, 20], [img.width - 20, img.height - 10], [20, img.height - 20]]
processed_img = F.perspective(img, startpoints, endpoints, interpolation=transforms.InterpolationMode.BILINEAR, fill=[0, 0, 0])

# Paddle 写法
import paddle
from PIL import Image
img = Image.open('path_to_image.jpg')
startpoints = [[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]
    ]
endpoints = [[10, 10], [img.width - 10, 20], [img.width - 20, img.height -
    10], [20, img.height - 20]]
processed_img = paddle.vision.transforms.perspective(img=img, startpoints=
    startpoints, endpoints=endpoints, interpolation='bilinear', fill=[0, 0, 0])

```
