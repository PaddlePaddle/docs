## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.perspective

### [torchvision.transforms.functional.perspective](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.perspective.html#torchvision.transforms.functional.perspective)

```python
torchvision.transforms.functional.perspective(img, startpoints, endpoints, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, fill=None)
```

### [paddle.vision.transforms.perspective](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/perspective_cn.html#paddle/vision/transforms/perspective_cn#cn-api-paddle-vision-transforms-perspective)

```python
paddle.vision.transforms.perspective(img, startpoints, endpoints, interpolation='nearest', fill=0)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.perspective(
    img, startpoints, endpoints, torchvision.transforms.InterpolationMode.BILINEAR, fill
)

# Paddle 写法
result = paddle.vision.transforms.perspective(
    img, startpoints, endpoints, torchvision.transforms.InterpolationMode.BILINEAR, fill
)
```
