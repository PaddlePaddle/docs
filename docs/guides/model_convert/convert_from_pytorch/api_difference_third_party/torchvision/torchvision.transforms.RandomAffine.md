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

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision        | PaddlePaddle    | 备注                      |
| ------------------ | ---------------- | ------------------------ |
| degrees           | degrees           | 随机旋转变换的角度大小。 |
| translate         | translate         | 随机水平平移和垂直平移变化的位移大小。 |
| scale             | scale             | 随机伸缩变换的比例大小。  |
| shear             | shear             | 随机剪切角度的大小范围。  |
| interpolation     | interpolation     | 插值的方法，PyTorch 参数为 InterpolationMode, Paddle 参数为 int 或 str 的形式，需要转写。|
| fill              | fill              | 对图像扩展时填充的像素值。   |
| center            | center            | 仿射变换的中心点坐标。   |
| -                 | keys              | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |


### 转写示例
#### interpolation：插值的方法

```python
# PyTorch 写法
transform = torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.2), scale=(0.8, 1.2), shear=10, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0, center=(100, 100))
transformed_img = transform(img)

# Paddle 写法
transform = paddle.vision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.2), scale=(0.8, 1.2), shear=10, interpolation='bilinear', fill=0, center=(100, 100))
transformed_img = transform(img)
```
