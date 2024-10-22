## [参数完全一致]torchvision.transforms.functional.adjust_hue

### [torchvision.transforms.functional.adjust_hue](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html)

```python
torchvision.transforms.functional.adjust_hue(img: Tensor, hue_factor: float)
```

### [paddle.vision.transforms.adjust_hue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_hue_cn.html)

```python
paddle.vision.transforms.adjust_hue(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    hue_factor: float
)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision           | PaddlePaddle      | 备注                                     |
| --------------------- | ----------------- | ---------------------------------------- |
| img                   | img               | 输入的图像。                              |
| hue_factor            | hue_factor        | 调整色调的因子。       |
