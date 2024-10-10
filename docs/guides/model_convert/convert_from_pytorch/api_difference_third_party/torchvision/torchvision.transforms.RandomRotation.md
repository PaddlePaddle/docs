## [输入参数类型不一致]torchvision.transforms.RandomRotation

### [torchvision.transforms.RandomRotation](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html)

```python
torchvision.transforms.RandomRotation(
    degrees: Union[int, List[float], Tuple[float, ...]],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[Union[List[float], Tuple[float, ...]]] = None,
    fill: Union[int, float, Tuple[int, ...]] = 0
)
```

### [paddle.vision.transforms.RandomRotation](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomRotation_cn.html)

```python
paddle.vision.transforms.RandomRotation(
    degrees: Union[int, List[float], Tuple[float, ...]],
    interpolation: Union[str, int] = 'nearest',
    expand: bool = False,
    center: Optional[Tuple[int, int]] = None,
    fill: int = 0,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| degrees              | degrees            | 旋转角度范围。                   |
| interpolation      | interpolation               | 参数名相同但类型不同，Paddle 使用 String 表示插值方法，转写时需要把 InterpolationMode 转写为 String。           |
| expand                | expand                   | 是否扩展图像尺寸。                    |
| center            | center        |  旋转的中心点坐标。             |
| fill           | fill                            | 对图像扩展时填充的值。               |
| -                                     | keys  | Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。 |

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
