## [输入参数类型不一致]torchvision.transforms.functional.affine

### [torchvision.transforms.functional.affine](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html)

```python
torchvision.transforms.functional.affine(img: Tensor,
                                        angle: float,
                                        translate: List[int],
                                        scale: float,
                                        shear: List[float],
                                        interpolation: InterpolationMode = InterpolationMode.NEAREST,
                                        fill: Optional[List[float]] = None,
                                        center: Optional[List[int]] = None)
```

### [paddle.vision.transforms.affine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/affine_cn.html)

```python
paddle.vision.transforms.affine(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    angle: Union[float, int],
    translate: List[float],
    scale: float,
    shear: Union[List[float], Tuple[float, ...]],
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    center: Optional[Tuple[int, int]] = None
)

```

两者功能基本一致，但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ----------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| img                               | img  | 输入图片。 |
| angle                              | angle              | 旋转角度。 |
| translate                      | translate           | 随机水平平移和垂直平移变化的位移大小。       |
| scale                              | scale                      | 控制缩放比例。                             |
| shear                        | shear             | 剪切角度值 。   |
| interpolation   | interpolation        | 参数类型不同，Paddle 使用 String 表示插值方法，转写时需要把 InterpolationMode 转写为 String  。          |
| fill     | fill       | 对图像扩展时填充的像素值。    |
| center               | center  |  仿射变换的中心点坐标 。    |

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
