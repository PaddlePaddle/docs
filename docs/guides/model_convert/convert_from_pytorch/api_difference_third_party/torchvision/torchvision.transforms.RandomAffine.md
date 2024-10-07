## [输入参数类型不一致] torchvision.transforms.RandomAffine

### [torchvision.transforms.RandomAffine](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html)

```python
torchvision.transforms.RandomAffine(
    degrees: Union[List[float], Tuple[float, ...], float],
    translate: Optional[Tuple[float, float]] = None,
    scale: Optional[Tuple[float, float]] = None,
    shear: Union[List[float], Tuple[float, ...], float] = None,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Union[int, float, List[float], Tuple[float, ...]] = 0,
    center: Optional[Union[List[int], Tuple[int, ...]]] = None
)
```

### [paddle.vision.transforms.RandomAffine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomAffine_cn.html)

```python
paddle.vision.transforms.RandomAffine(
    degrees: Union[Tuple[float, float], float, int],
    translate: Optional[Union[Sequence[float], float, int]] = None,
    scale: Optional[Tuple[float, float]] = None,
    shear: Optional[Union[Sequence[float], float, int]] = None,
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    center: Optional[Tuple[int, int]] = None,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能基本一致，但参数类型存在不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle    | 备注                                                         |
| ------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| degrees                | degrees  | 随机旋转变换的角度大小。 |
| translate                 | translate  | 随机水平平移和垂直平移变化的位移大小。 |
| scale                     | scale      | 随机伸缩变换的比例大小。                                   |
| shear        | shear    | 随机剪切角度的大小范围。                           |
| interpolation | interpolation                 | 参数类型不同，Paddle 使用字符串或整数表示插值方法。            |
| fill             | fill      |  对图像扩展时填充的像素值，默认值： 0 。                   |
| center                 | center         | 仿射变换的中心点坐标。   |
| -                                          | keys   | Paddle 支持 `keys` 参数，torchvision 不支持。                 |


### 转写示例


```python
# PyTorch 写法
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.RandomAffine(degrees=30,
                                    translate=(0.1, 0.2),
                                    scale=(0.8, 1.2),
                                    shear=10,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=0,
                                    center=(100, 100))
img = Image.open('path_to_image.jpg')
transformed_img = transform(img)

# Paddle 写法
import paddle
from PIL import Image
transform = paddle.vision.transforms.RandomAffine(degrees=30, translate=(
    0.1, 0.2), scale=(0.8, 1.2), shear=10, interpolation='bilinear', fill=0,
    center=(100, 100))
img = Image.open('path_to_image.jpg')
transformed_img = transform(img)
```
