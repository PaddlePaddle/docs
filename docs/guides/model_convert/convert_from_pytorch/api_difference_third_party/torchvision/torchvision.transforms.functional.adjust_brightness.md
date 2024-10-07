## [参数完全一致]torchvision.transforms.functional.adjust_brightness

### [torchvision.transforms.functional.adjust_brightness](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_brightness.html)

```python
torchvision.transforms.functional.adjust_brightness(img: Tensor, brightness_factor: float)
```

### [paddle.vision.transforms.adjust_brightness](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_brightness_cn.html)

```python
paddle.vision.transforms.adjust_brightness(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor], 
    brightness_factor: float
)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                     |
| ----------------------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| img                              | img | Paddle 支持更多类型，但包含 torch 支持的类型。 |
| brightness_factor                              | brightness_factor                    | 调整亮度的因子。                          |
