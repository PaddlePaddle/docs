## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.perspective
### [torchvision.transforms.functional.perspective](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.perspective.html#perspective)
```python
torchvision.transforms.functional.perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None
)
```

### [paddle.vision.transforms.perspective](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/perspective_cn.html#cn-api-paddle-vision-transforms-perspective)
```python
paddle.vision.transforms.perspective(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    startpoints: List[List[float]],
    endpoints: List[List[float]],
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0
)
```

两者功能一致，但参数类型不一致。
