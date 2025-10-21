## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.affine
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
