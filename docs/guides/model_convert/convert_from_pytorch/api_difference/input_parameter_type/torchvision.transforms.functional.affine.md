## [输入参数类型不一致]torchvision.transforms.functional.affine

### [torchvision.transforms.functional.affine](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html)

```python
torchvision.transforms.functional.affine(img: Tensor,
                                        angle: float,
                                        translate: List[int],
                                        scale: float,
                                        shear: List[float],
                                        interpolation: InterpolationMode = InterpolationMode.NEAREST,
                                        fill: Optional[List[float]] = None,
                                        center: Optional[List[int]] = None)
```

### [paddle.vision.transforms.affine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/affine_cn.html)

```python
paddle.vision.transforms.affine(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    angle: Union[float, int],
    translate: List[float],
    scale: float,
    shear: Union[List[float], Tuple[float, ...]],
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    center: Optional[Tuple[int, int]] = None
)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ----------- | ------------ | ------------------------------------------------------------ |
| img               | img           | 输入图片。 |
| angle             | angle         | 旋转角度。 |
| translate         | translate     | 随机水平平移和垂直平移变化的位移大小。       |
| scale             | scale         | 控制缩放比例。                             |
| shear             | shear         | 剪切角度值。   |
| interpolation     | interpolation | 插值的方法，两者类型不一致，PyTorch 为 InterpolationMode 枚举类, Paddle 为 int 或 string，需要转写。         |
| fill              | fill          | 对图像扩展时填充的像素值。    |
| center            | center        | 仿射变换的中心点坐标 。    |

### 转写示例
#### interpolation：插值的方法
```python
# PyTorch 写法
rotated_img = torchvision.transforms.functional.affine(img, angle=30.0, translate=[10, 20], scale=1.2, shear=[10.0, 5.0], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=[0, 0, 0], center=[100, 100])

# Paddle 写法
rotated_img = paddle.vision.transforms.affine(img=img, angle=30.0, translate=[10, 20], scale=1.2, shear=[10.0, 5.0], interpolation='bilinear', fill=[0, 0, 0], center=[100, 100])
```
