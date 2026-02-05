## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.affine

### [torchvision.transforms.functional.affine](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.affine.html#torchvision.transforms.functional.affine)

```python
torchvision.transforms.functional.affine(img, angle, translate, scale, shear, interpolation=torchvision.transforms.functional.NEREAST, fill=None, center=None)
```

### [paddle.vision.transforms.affine](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/affine_cn.html#paddle.vision.transforms.affine)

```python
paddle.vision.transforms.affine(img, angle, translate, scale, shear, interpolation='nearst', fill=0, center=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.affine(
    img, 30, (1, 1), 1.0, 0, torchvision.transforms.InterpolationMode.NEAREST
)

# Paddle 写法
result = paddle.vision.transforms.affine(
    img, 30, (1, 1), 1.0, 0, torchvision.transforms.InterpolationMode.NEAREST
)

```
