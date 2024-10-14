## [输入参数类型不一致]torchvision.transforms.functional.resize

### [torchvision.transforms.functional.resize](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html)

```python
torchvision.transforms.functional.resize(img: Tensor, size: Optional[Union[int, Sequence[int]]],
                             interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
                             max_size: Optional[int] = None,
                             antialias: Optional[bool] = True)
```

### [paddle.vision.transforms.resize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/resize_cn.html)

```python
paddle.vision.transforms.resize(
    img: Union[np.ndarray, paddle.Tensor, PIL.Image.Image],
    size: Union[int, List[int], Tuple[int, ...]],
    interpolation: Union[str, int] = 'bilinear',
)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------ | ------------ | ---------------- |
| img            | img              | 输入数据。         |
| size           | size             | 输出图像大小。         |
| interpolation  | interpolation    | 插值的方法，PyTorch 参数为 InterpolationMode, Paddle 参数为 int 或 str 的形式，需要转写。          |
| max_size       | -                | 允许的最长边的最大值，Paddle 无此参数，暂无转写方式。                             |
| antialias      | -                | 是否应用抗锯齿处理，Paddle 无此参数，暂无转写方式。                             |


### 转写示例
#### interpolation：插值的方法

```python
# PyTorch 写法
resized_img = torchvision.transforms.functional.resize(img, size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

# Paddle 写法
resized_img = paddle.vision.transforms.resize(img=img, size=(224, 224), interpolation='bilinear')
```
