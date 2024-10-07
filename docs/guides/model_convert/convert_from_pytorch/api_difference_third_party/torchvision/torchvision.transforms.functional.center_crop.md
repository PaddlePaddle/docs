## [参数完全一致]torchvision.transforms.functional.center_crop

### [torchvision.transforms.functional.center_crop](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.center_crop.html)

```python
torchvision.transforms.functional.center_crop(img: Tensor, output_size: List[int])
```

### [paddle.vision.transforms.center_crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/center_crop_cn.html)

```python
paddle.vision.transforms.center_crop(
    img: Union[PIL.Image.Image, paddle.Tensor, np.ndarray], 
    output_size: List[int]
)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                             |
| ----------------------------------------------- | ------------------------------------- | -------------------------------- |
| img                        | img            | 输入图片。       |
| output_size                   | output_size         | 输出尺寸。 |
