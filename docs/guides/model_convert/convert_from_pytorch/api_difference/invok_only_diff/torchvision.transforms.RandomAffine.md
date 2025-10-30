## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomAffine

### [torchvision.transforms.RandomAffine](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine)

```python
torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=torchvision.transforms.InterpolationMode.InterpolationMode.NEAREST, fill=0, center=None)
```

### [paddle.vision.transforms.RandomAffine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomAffine_cn.html#paddle/vision/transforms/RandomAffine_cn#cn-api-paddle-vision-transforms-RandomAffine)

```python
paddle.vision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation='nearest', fill=0, center=None, keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
transform = torchvision.transforms.RandomAffine(
    45,
    (0.1, 0.1),
    (0.5, 1.5),
    20,
    torchvision.transforms.InterpolationMode.BILINEAR,
    255,
    (2, 2),
)

# Paddle 写法
transform = paddle.vision.transforms.RandomAffine(
    45,
    (0.1, 0.1),
    (0.5, 1.5),
    20,
    "bilinear",
    255,
    (2, 2)
)

```
