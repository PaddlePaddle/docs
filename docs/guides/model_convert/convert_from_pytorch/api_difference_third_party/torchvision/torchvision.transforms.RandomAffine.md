## [输入参数类型不一致] torchvision.transforms.RandomAffine

### [torchvision.transforms.RandomAffine](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html)

```python
torchvision.transforms.RandomAffine(degrees: sequence | number,
                                    translate: Optional[Tuple[float, float]] = None,
                                    scale: Optional[Tuple[float, float]] = None,
                                    shear: sequence | number = None,
                                    interpolation: InterpolationMode = InterpolationMode.NEAREST,
                                    fill: int | float | sequence = 0,
                                    center: Optional[Sequence[int]] = None)
```

### [paddle.vision.transforms.RandomAffine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomAffine_cn.html)

```python
paddle.vision.transforms.RandomAffine(degrees: Tuple[float, float] | float | int,
                                      translate: Sequence[float] | float | int = None,
                                      scale: Tuple[float, float] = None,
                                      shear: Sequence[float] | float | int = None,
                                      interpolation: str | int = 'nearest',
                                      fill: int | List[int] | Tuple[int] = 0,
                                      center: Tuple[int, int] = None,
                                      keys: List[str] | Tuple[str] = None)
```

两者功能基本一致，但参数类型存在不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle    | 备注                                                         |
| ------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| degrees (Tuple[float, float] or float or int)               | degrees (Tuple[float, float] or float or int) | 随机旋转变换的角度大小。 |
| translate (tuple, optional)                | translate (Sequence[float] or float or int) | 随机水平平移和垂直平移变化的位移大小。 |
| scale (tuple, optional)                    | scale (Tuple[float, float], optional)     | 随机伸缩变换的比例大小。                                   |
| shear (sequence or number, optional)       | shear (Sequence[float] or float or int)   | 随机剪切角度的大小范围。                           |
| interpolation (InterpolationMode, optional)| interpolation (str or int)                | 参数类型不同，Paddle 使用字符串或整数表示插值方法。            |
| fill (int or float or sequence)            | fill (int or List[int] or Tuple[int])     |  对图像扩展时填充的像素值，默认值： 0 。                   |
| center (sequence, optional)                | center (Tuple[int, int], optional)        | 仿射变换的中心点坐标。   |
| -                                          | keys (List[str] or Tuple[str], optional)  | Paddle 支持 `keys` 参数，torchvision 不支持。                 |


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
