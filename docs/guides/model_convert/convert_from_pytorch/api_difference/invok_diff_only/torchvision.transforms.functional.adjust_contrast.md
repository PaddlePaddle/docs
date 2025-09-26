## [参数完全一致]torchvision.transforms.functional.adjust_contrast

### [torchvision.transforms.functional.adjust_contrast](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html)

```python
torchvision.transforms.functional.adjust_contrast(img: Tensor, contrast_factor: float)
```

### [paddle.vision.transforms.adjust_contrast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_contrast_cn.html)

```python
paddle.vision.transforms.adjust_contrast(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    contrast_factor: float
)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision           | PaddlePaddle      | 备注                                     |
| --------------------- | ----------------- | ---------------------------------------- |
| img                   | img               | 输入的图像。           |
| contrast_factor       | contrast_factor   | 调整对比度的因子。                          |
