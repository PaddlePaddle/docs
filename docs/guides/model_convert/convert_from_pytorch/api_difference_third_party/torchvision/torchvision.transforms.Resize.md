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
| size  | size       | 输出图像大小。                       |
| interpolation  | interpolation  | 参数名相同但类型不同，Paddle 使用 String 表示插值方法，转写时需要把 InterpolationMode 转写为 String。            |
| max_size         | -                                | Paddle 无此参数，暂无转写方式。                          |
| antialias       | -                                | Paddle 无此参数，暂无转写方式。                           |
| -                             | keys  | Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。 |


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
