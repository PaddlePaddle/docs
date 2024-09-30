## [输入参数类型不一致]torchvision.transforms.RandomPerspective

### [torchvision.transforms.RandomPerspective](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomPerspective.html?highlight=randomperspective#torchvision.transforms.RandomPerspective)

```python
torchvision.transforms.RandomPerspective(distortion_scale: float = 0.5,
                                        p: float = 0.5,
                                        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                                        fill: int | float | sequence = 0)
```

### [paddle.vision.transforms.RandomPerspective](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomPerspective_cn.html)

```python
paddle.vision.transforms.RandomPerspective(prob: float = 0.5,
                                           distortion_scale: float = 0.5,
                                           interpolation: str | int = 'nearest',
                                           fill: int | list | tuple = 0,
                                           keys: list[str] | tuple[str] = None)
```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ----------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| distortion_scale (float)                  | distortion_scale (float)                   | 两者参数名称和功能一致，控制失真程度。                       |
| p (float)                                 | prob (float)                               | 参数名不同，Paddle 使用 `prob` 替代 `p`，表示透视变换的概率。 |
| interpolation (InterpolationMode or int)  | interpolation (str or int)                | 参数名相同但类型不同，Paddle 使用字符串或整数表示插值方法。    |
| fill (int or list or tuple)           | fill (int or list or tuple)                | 对图像扩展时填充的值。默认值： 0。                 |
| -                                         | keys (list[str] or tuple[str])             | Paddle 支持 `keys` 参数。 |

### 转写示例

```python
# PyTorch 写法
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.RandomPerspective(distortion_scale=0.5,
                                        p=0.5,
                                        interpolation=transforms.InterpolationMode.BILINEAR,
                                        fill=0)
img = Image.open('path_to_image.jpg')
transformed_img = transform(img)

# Paddle 写法
import paddle
from PIL import Image
transform = paddle.vision.transforms.RandomPerspective(distortion_scale=0.5,
    prob=0.5, interpolation='bilinear', fill=0)
img = Image.open('path_to_image.jpg')
transformed_img = transform(img)
```
