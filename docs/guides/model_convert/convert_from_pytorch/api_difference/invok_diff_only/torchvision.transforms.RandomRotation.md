## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomRotation
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
