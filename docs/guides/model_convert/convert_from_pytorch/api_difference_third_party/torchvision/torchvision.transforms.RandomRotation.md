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
| ------------------- | ------------------ | ----------------------------------------------- |
| degrees               | degrees           | 旋转角度范围。                   |
| interpolation         | interpolation     | 插值的方法，PyTorch 参数为 InterpolationMode, Paddle 参数为 int 或 str 的形式，需要转写。|
| expand                | expand            | 是否扩展图像尺寸。                    |
| center                | center            | 旋转的中心点坐标。             |
| fill                  | fill              | 对图像扩展时填充的值。               |
| -                     | keys              | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |

### 转写示例
#### interpolation：插值的方法
```python
# PyTorch 写法
transform = torchvision.transforms.RandomRotation(degrees=45, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=True, center=(100, 100), fill=(255, 0, 0))
rotated_img = transform(img)

# Paddle 写法
transform = paddle.vision.transforms.RandomRotation(degrees=45, interpolation='bilinear', expand=True, center=(100, 100), fill=(255, 0, 0))
rotated_img = transform(img)
```
