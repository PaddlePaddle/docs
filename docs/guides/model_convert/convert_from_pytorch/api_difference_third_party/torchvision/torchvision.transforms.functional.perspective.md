## [输入参数类型不一致]torchvision.transforms.functional.perspective

### [torchvision.transforms.functional.perspective](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.perspective.html#perspective)

```python
torchvision.transforms.functional.perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None
)
```

### [paddle.vision.transforms.perspective](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/perspective_cn.html#cn-api-paddle-vision-transforms-perspective)

```python
paddle.vision.transforms.perspective(
    img: Union[PIL.Image.Image, np.ndarray, paddle.Tensor],
    startpoints: List[List[float]],
    endpoints: List[List[float]],
    interpolation: Union[str, int] = 'nearest',
    fill: Union[int, List[int], Tuple[int, ...]] = 0
)
```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision       | PaddlePaddle | 备注              |
| ----------------- | ----------------- | ------------ |
| img               | img               | 输入图片。    |
| startpoints       | startpoints       | 在原图上的四个角（左上、右上、右下、左下）的坐标。      |
| endpoints         | endpoints         | 在原图上的四个角（左上、右上、右下、左下）的坐标。      |
| interpolation     | interpolation     | 插值的方法，PyTorch 参数为 InterpolationMode, Paddle 参数为 int 或 str 的形式，需要转写。          |
| fill              | fill              | 对图像扩展时填充的像素值。             |


### 转写示例
#### interpolation：插值的方法
```python
# PyTorch 写法
processed_img = torchvision.transforms.functional.perspective(img, startpoints, endpoints, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=[0, 0, 0])

# Paddle 写法
processed_img = paddle.vision.transforms.perspective(img=img, startpoints=startpoints, endpoints=endpoints, interpolation='bilinear', fill=[0, 0, 0])
```
