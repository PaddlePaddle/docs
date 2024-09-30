## [参数完全一致]torchvision.transforms.functional.crop

### [torchvision.transforms.functional.crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html)

```python
torchvision.transforms.functional.crop(img: PIL Image | Tensor, top: int, left: int, height: int, width: int)
```

### [paddle.vision.transforms.crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/crop_cn.html)

```python
paddle.vision.transforms.crop(img: PIL.Image | np.array, top: int, left: int, height: int, width: int)
```

两者功能一致，输入参数类型一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| img (PIL Image or Tensor)             | img (PIL.Image or np.array or paddle.Tensor)    | 输入图片。 |
| top (int) | top (int) | 要裁剪的矩形框左上方的坐标点的垂直方向的值。 |
| left (int)| left (int)| 要裁剪的矩形框左上方的坐标点的水平方向的值。 |
| height (int)| height (int)| 要裁剪的矩形框的高度值。 |
| width (int)| width (int)| 要裁剪的矩形框的宽度值。 |
