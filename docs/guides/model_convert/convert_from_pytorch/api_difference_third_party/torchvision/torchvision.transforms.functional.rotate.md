## [输入参数类型不一致]torchvision.transforms.functional.rotate

### [torchvision.transforms.functional.rotate](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.rotate.html)

```python
torchvision.transforms.functional.rotate(img: Tensor, angle: float, interpolation: InterpolationMode = InterpolationMode.NEAREST, expand: bool = False, center: Optional[List[int]] = None, fill: Optional[List[float]] = None)
```

### [paddle.vision.transforms.rotate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/rotate_cn.html#cn-api-paddle-vision-transforms-rotate)

```python
paddle.vision.transforms.rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=0)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| img                                   | img                | 输入图片。|
| angle                                 | angle              | 旋转角度。|
| interpolation                         | interpolation      | 插值的方法，两者类型不一致，PyTorch 为 InterpolationMode 枚举类, Paddle 为 int 或 string，需要转写。         |
| expand                                | expand             | 是否要对旋转后的图片进行大小扩展。|
| center                                | center             | 旋转中心。|
| fill                                  | fill               | 旋转图像外部区域的 RGB 像素填充值。|


### 转写示例
#### interpolation：插值的方法

```python
# PyTorch 写法
torchvision.transforms.functional.rotate(img, angle=90, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

# Paddle 写法
paddle.vision.transforms.rotate(img=img, angle=90, interpolation='bilinear')
```
