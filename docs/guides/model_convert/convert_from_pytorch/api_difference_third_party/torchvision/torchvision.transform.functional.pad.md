## [参数完全一致]torchvision.transforms.functional.pad

### [torchvision.transforms.functional.pad](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html)

```python
torchvision.transforms.functional.pad(
    img: Union[PIL.Image.Image, torch.Tensor],
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant'
)
```

### [paddle.vision.transforms.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/pad_cn.html)

```python
paddle.vision.transforms.pad(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant'
)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision      | PaddlePaddle| 备注                                                         |
| ---------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| img              | img            | 被填充的图像。|
| padding          | padding        | 在图像边界上进行填充的范围。                 |
| fill             | fill           | 用于填充的像素值。         |
| padding_mode     | padding_mode   | 填充模式。|
