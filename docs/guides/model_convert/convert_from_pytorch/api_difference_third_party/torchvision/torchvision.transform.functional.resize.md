## [输入参数类型不一致]torchvision.transforms.functional.resize

### [torchvision.transforms.resize](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html)

```python
torchvision.transforms.functional.resize(img: Tensor, size: Optional[Union[int, Sequence[int]]],
                             interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
                             max_size: Optional[int] = None,
                             antialias: Optional[bool] = True)
```

### [paddle.vision.transforms.resize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/resize_cn.html)

```python
paddle.vision.transforms.Resize(img: numpy.ndarray | Tensor | PIL.Image, size: int | list | tuple,
                                interpolation: str | int = 'bilinear',
                                keys: list[str] | tuple[str] = None)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | paddle | 备注                                                         |
| ----------------------------- | -------------------------------- | ------------------------------------------------------------ |
| size (int \| list \| tuple) | size (int \| list \| tuple)      | 两者均支持单个整数或序列表示输出大小。                       |
| interpolation (InterpolationMode) | interpolation (str \| int) | 两者类型不一致。               |
| max_size (int \| None)        | -                                | Paddle 不支持 `max_size` 参数。                             |
| antialias (bool \| None)      | -                                | Paddle 不支持 `antialias` 参数。                             |
| -                             | keys (list[str] \| tuple[str] = None) | Paddle 支持 `keys` 参数。 |


### 转写示例


```python
# PyTorch 写法
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
img = Image.open('path_to_image.jpg')
resized_img = F.resize(img, size=(224, 224),
                             interpolation=transforms.InterpolationMode.BILINEAR)

# Paddle 写法
import paddle
from PIL import Image
img = Image.open('path_to_image.jpg')
resized_img = paddle.vision.transforms.resize(img=img, size=(224, 224),
    interpolation='bilinear')

```
