## [输入参数类型不一致]torchvision.transforms.functional.resize

### [torchvision.transforms.functional.resize](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html)

```python
torchvision.transforms.functional.resize(img: Tensor, size: Optional[Union[int, Sequence[int]]],
                             interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
                             max_size: Optional[int] = None,
                             antialias: Optional[bool] = True)
```

### [paddle.vision.transforms.resize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/resize_cn.html)

```python
paddle.vision.transforms.resize(
    img: Union[np.ndarray, paddle.Tensor, PIL.Image.Image],
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
| interpolation  | interpolation  | 两者类型不一致，torch 为 InterpolationMode，转写时需要把 InterpolationMode 转写为 String。               |
| max_size         | -                                | 表示调整图像大小时允许的最长边的最大值，Paddle 无此参数，暂无转写方式。                             |
| antialias       | -                                | 是否应用抗锯齿处理，Paddle 无此参数，暂无转写方式。                             |
| -                             | keys  | Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。 |


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
