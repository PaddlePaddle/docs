## [输入参数类型不一致]torchvision.transforms.functional.affine

### [torchvision.transforms.functional.affine](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html?highlight=affine#torchvision.transforms.functional.affine)

```python
torchvision.transforms.functional.affine(img: Tensor,
                                        angle: float,
                                        translate: List[int],
                                        scale: float,
                                        shear: List[float],
                                        interpolation: InterpolationMode = InterpolationMode.NEAREST,
                                        fill: Optional[List[float]] = None,
                                        center: Optional[List[int]] = None) → Tensor
```

### [paddle.vision.transforms.affine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/affine_cn.html)

```python
paddle.vision.transforms.affine(img: PIL.Image | numpy.ndarray | paddle.Tensor,
                                angle: float | int,
                                translate: list[float],
                                scale: float,
                                shear: list | tuple,
                                interpolation: str | int = 'nearest',
                                fill: int | list | tuple = 0,
                                center: tuple[int, int] = None) → PIL.Image | numpy.ndarray | paddle.Tensor
```

两者功能基本一致，但参数类型不一致，具体如下：

### 参数映射

| torchvision.transforms.functional.affine | paddle.vision.transforms.affine | 备注                                                         |
| ----------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| img (Tensor)                              | img (PIL.Image \| numpy.ndarray \| paddle.Tensor) | 输入图片 |
| angle (float)                             | angle (float \| int)             | Paddle 类型更灵活 |
| translate (List[int])                     | translate (list[float])          | 数据类型不同，torchvision 使用整数，Paddle 使用浮点数       |
| scale (float)                             | scale (float)                     | 参数名称和功能一致，控制缩放比例                             |
| shear (list)                       | shear (list \| tuple)            | 剪切角度值    |
| interpolation (InterpolationMode \| int)  | interpolation (str \| int)       | 参数类型不同，Paddle 使用字符串或整数表示插值方法            |
| fill (int \| list \| tuple)    | fill (int \| list \| tuple)      | 对图像扩展时填充的像素值，默认值：0       |
| center (tuple[int, int] \| None)              | center (tuple[int, int] \| None) |  仿射变换的中心点坐标     |

### 转写示例

```python
# PyTorch 写法
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
img = Image.open('path_to_image.jpg')
angle = 30.0
translate = [10, 20]
scale = 1.2
shear = [10.0, 5.0]
rotated_img = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=transforms.InterpolationMode.BILINEAR, fill=[0, 0, 0], center=[100, 100])

# Paddle 写法
import paddle
from PIL import Image
img = Image.open('path_to_image.jpg')
angle = 30.0
translate = [10, 20]
scale = 1.2
shear = [10.0, 5.0]
rotated_img = paddle.vision.transforms.affine(img=img, angle=angle,
    translate=translate, scale=scale, shear=shear, interpolation='bilinear',
    fill=[0, 0, 0], center=[100, 100])

```
