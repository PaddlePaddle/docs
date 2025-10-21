## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomAffine
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
