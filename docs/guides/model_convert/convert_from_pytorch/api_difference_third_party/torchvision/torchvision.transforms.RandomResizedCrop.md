## [输入参数类型不一致]torchvision.transforms.RandomResizedCrop

### [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)

```python
torchvision.transforms.RandomResizedCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = True
)
```

### [paddle.vision.transforms.RandomResizedCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomResizedCrop_cn.html)

```python
paddle.vision.transforms.RandomResizedCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    scale: Union[List[float], Tuple[float, float]] = (0.08, 1.0),
    ratio: Union[List[float], Tuple[float, float]] = (0.75, 1.33),
    interpolation: Union[int, str] = 'bilinear',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| size                      | size                  | 裁剪后的图片大小。                                           |
| scale                      | scale                         | 相对于原图的尺寸，随机裁剪后图像大小的范围。                |
| ratio                      | ratio                         | 裁剪后的目标图像宽高比范围。                                 |
| interpolation           | interpolation                   | Paddle 支持更多插值方法，接受整数或字符串形式。               |
| antialias                  | -                                           | Paddle 不支持 `antialias` 参数。                             |
| -                                          | keys         | Paddle 支持 `keys` 参数，可用于指定要裁剪的键。              |

### 转写示例

```python
# PyTorch 写法
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BILINEAR)
img = Image.open('path_to_image.jpg')
cropped_img = transform(img)

# Paddle 写法
import paddle
from PIL import Image
transform = paddle.vision.transforms.RandomResizedCrop(size=(224, 224),
    scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation='bilinear')
img = Image.open('path_to_image.jpg')
cropped_img = transform(img)

```
