## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.rotate
### [torchvision.transforms.functional.rotate](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html)
```python
torchvision.transforms.functional.rotate(img: Tensor, angle: float, interpolation: InterpolationMode = InterpolationMode.NEAREST, expand: bool = False, center: Optional[List[int]] = None, fill: Optional[List[float]] = None)
```

### [paddle.vision.transforms.rotate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/rotate_cn.html#cn-api-paddle-vision-transforms-rotate)
```python
paddle.vision.transforms.rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=0)
```

两者功能一致，但输入参数类型不一致。
