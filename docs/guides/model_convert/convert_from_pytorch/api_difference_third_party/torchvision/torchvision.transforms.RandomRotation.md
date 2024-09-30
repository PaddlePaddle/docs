## [输入参数类型不一致]torchvision.transforms.RandomRotation

### [torchvision.transforms.RandomRotation](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html)

```python
torchvision.transforms.RandomRotation(degrees: int | sequence,
                                      interpolation: InterpolationMode = InterpolationMode.NEAREST,
                                      expand: bool = False,
                                      center: Optional[sequence] = None,
                                      fill: int | float | tuple = 0)
```

### [paddle.vision.transforms.RandomRotation](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomRotation_cn.html)

```python
paddle.vision.transforms.RandomRotation(degrees: int | list | tuple,
                                        interpolation: str | int = 'nearest',
                                        expand: bool = False,
                                        center: tuple[int, int] = None,
                                        fill: int = 0,
                                        keys: list[str] | tuple[str] = None)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| degrees (int or list or tuple)             | degrees (int or list or tuple)           | 两者均支持单个整数或序列表示旋转角度范围。                   |
| interpolation (InterpolationMode)     | interpolation (str or int)              | 支持的插值方法。               |
| expand (bool, optional)               | expand (bool, optional)                  | 两者均支持是否扩展图像尺寸。                                 |
| center (int or list or tuple, optional)           | center (tuple[int, int], optional)       | Paddle 的 center 参数仅支持整数坐标。                       |
| fill (int or float or tuple)          | fill (int or float or tuple)                           | 对图像扩展时填充的值。默认值：0。               |
| -                                     | keys (list[str] or tuple[str], optional) | Paddle 支持 `keys` 参数，用于指定要旋转的键，torchvision 不支持。 |

### 转写示例

```python
# PyTorch 写法
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.RandomRotation(degrees=45, interpolation=transforms.InterpolationMode.BILINEAR, expand=True, center=(100, 100), fill=(255, 0, 0))
img = Image.open('path_to_image.jpg')
rotated_img = transform(img)

# Paddle 写法
import paddle
from PIL import Image
transform = paddle.vision.transforms.RandomRotation(degrees=45,
    interpolation='bilinear', expand=True, center=(100, 100), fill=(255, 0, 0))
img = Image.open('path_to_image.jpg')
rotated_img = transform(img)
```
