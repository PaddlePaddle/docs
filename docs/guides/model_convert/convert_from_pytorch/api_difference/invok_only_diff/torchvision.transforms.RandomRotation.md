## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomRotation

### [torchvision.transforms.RandomRotation](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomRotation.html#torchvision.transforms.RandomRotation)

```python
torchvision.transforms.RandomRotation(degrees, interpolation=torchvision.transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0)
```

### [paddle.vision.transforms.RandomRotation](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomRotation_cn.html#paddle.vision.transforms.RandomRotation)

```python
paddle.vision.transforms.RandomRotation(degrees, interpolation='nearest', expand=False, center=None, fill=0, keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
rotation = torchvision.transforms.RandomRotation(degrees=degrees)

# Paddle 写法
rotation = paddle.vision.transforms.RandomRotation(degrees=degrees)
```
