## [参数完全一致]torchvision.transforms.functional.adjust_contrast

### [torchvision.transforms.functional.adjust_contrast](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html)

```python
torchvision.transforms.functional.adjust_contrast(img: Tensor, contrast_factor: float)
```

### [paddle.vision.transforms.adjust_contrast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_contrast_cn.html)

```python
paddle.vision.transforms.adjust_contrast(img: PIL.Image|np.array|paddle.Tensor, contrast_factor: float)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision | paddle | 备注                                     |
| --------------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| img (PIL Image or Tensor)                           | img (PIL.Image \| np.array \| paddle.Tensor) | Paddle 支持更多类型，但包含 torchvision 支持的类型。 |
| contrast_factor (float)                             | contrast_factor (float)                  | 调整对比度的因子。                          |
