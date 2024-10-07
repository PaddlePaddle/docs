## [参数完全一致]torchvision.transforms.functional.crop

### [torchvision.transforms.functional.crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html)

```python
torchvision.transforms.functional.crop(
    img: Union[PIL.Image.Image, torch.Tensor], 
    top: int, 
    left: int, 
    height: int, 
    width: int
)
```

### [paddle.vision.transforms.crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/crop_cn.html)

```python
paddle.vision.transforms.crop(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor], 
    top: int, 
    left: int, 
    height: int, 
    width: int
)
```

两者功能一致，输入参数类型一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| img              | img     | 输入图片。 |
| top  | top  | 要裁剪的矩形框左上方的坐标点的垂直方向的值。 |
| left | left | 要裁剪的矩形框左上方的坐标点的水平方向的值。 |
| height | height | 要裁剪的矩形框的高度值。 |
| width | width | 要裁剪的矩形框的宽度值。 |
