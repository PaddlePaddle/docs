## [输入参数类型不一致]torchvision.transforms.RandomResizedCrop

### [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)

```python
torchvision.transforms.RandomResizedCrop(size: int | list | tuple, scale: tuple = (0.08, 1.0), ratio: tuple = (0.75, 1.3333333333333333), interpolation: InterpolationMode = InterpolationMode.BILINEAR, antialias: Optional[bool] = True)
```

### [paddle.vision.transforms.RandomResizedCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomResizedCrop_cn.html)

```python
paddle.vision.transforms.RandomResizedCrop(size: int | list | tuple, scale: list | tuple = (0.08, 1.0), ratio: list | tuple = (0.75, 1.33), interpolation: int | str = 'bilinear', keys: list[str] | tuple[str] = None)
```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision | paddle | 备注                                                         |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| size (int \| list \| tuple)                     | size (int \| list \| tuple)                 | 裁剪后的图片大小。                                           |
| scale (tuple of float)                     | scale (list \| tuple)                        | 相对于原图的尺寸，随机裁剪后图像大小的范围。                |
| ratio (tuple of float)                     | ratio (list \| tuple)                        | 裁剪后的目标图像宽高比范围。                                 |
| interpolation (InterpolationMode)          | interpolation (int \| str)                  | Paddle 支持更多插值方法，接受整数或字符串形式。               |
| antialias (bool, optional)                 | -                                           | Paddle 不支持 `antialias` 参数。                             |
| -                                          | keys (list[str] \| tuple[str] = None)        | Paddle 支持 `keys` 参数，可用于指定要裁剪的键。              |

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
