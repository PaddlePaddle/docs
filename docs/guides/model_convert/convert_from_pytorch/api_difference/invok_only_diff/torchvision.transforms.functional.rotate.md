## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.rotate

### [torchvision.transforms.functional.rotate](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html#torchvision.transforms.functional.rotate)

```python
torchvision.transforms.functional.rotate(img, angle, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST, expand=False, center=None, fill=None)
```

### [paddle.vision.transforms.rotate](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/rotate_cn.html#paddle/vision/transforms/rotate_cn#cn-api-paddle-vision-transforms-rotate)

```python
paddle.vision.transforms.rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=0)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.rotate(
    img=img,
    angle=30,
    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    expand=True,
    center=(1, 1),
    fill=[128, 128, 128],
)


# Paddle 写法
result = paddle.vision.transforms.rotate(
    img=img,
    angle=30,
    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    expand=True,
    center=(1, 1),
    fill=[128, 128, 128],
)
```
