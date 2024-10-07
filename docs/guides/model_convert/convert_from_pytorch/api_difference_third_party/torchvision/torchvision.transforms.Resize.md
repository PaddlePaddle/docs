## [输入参数类型不一致]torchvision.transforms.Resize

### [torchvision.transforms.Resize](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize)

```python
torchvision.transforms.Resize(
    size: Optional[Union[int, List[int], Tuple[int, ...]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True
)
```

### [paddle.vision.transforms.Resize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Resize__upper_cn.html#resize)

```python
paddle.vision.transforms.Resize(
    size: Union[int, List[int], Tuple[int, ...]],
    interpolation: Union[str, int] = 'bilinear',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ----------------------------- | -------------------------------- | ------------------------------------------------------------ |
| size  | size       | 两者均支持单个整数或序列表示输出大小。                       |
| interpolation  | interpolation  | 两者类型不一致。               |
| max_size         | -                                | Paddle 不支持 `max_size` 参数。                             |
| antialias       | -                                | Paddle 不支持 `antialias` 参数。                             |
| -                             | keys  | Paddle 支持 `keys` 参数。 |


### 转写示例


```python
# PyTorch 写法
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.Resize(size=(224, 224),
                             interpolation=transforms.InterpolationMode.BILINEAR)
img = Image.open('path_to_image.jpg')
resized_img = transform(img)

# Paddle 写法
import paddle
from PIL import Image
transform = paddle.vision.transforms.Resize(size=(224, 224), interpolation=
    'bilinear')
img = Image.open('path_to_image.jpg')
resized_img = transform(img)

```