## [输入参数类型不一致]torchvision.transforms.Resize

### [torchvision.transforms.Resize](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize)

```python
torchvision.transforms.Resize(
    size: Optional[Union[int, List[int], Tuple[int, ...]]],
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True
)
```

### [paddle.vision.transforms.Resize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Resize__upper_cn.html#resize)

```python
paddle.vision.transforms.Resize(
    size: Union[int, List[int], Tuple[int, ...]],
    interpolation: Union[str, int] = 'bilinear',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但输入参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ------------- | --------------- | ------------------------------------------------------------ |
| size           | size             | 输出图像大小。                       |
| interpolation  | interpolation    | 插值的方法，两者类型不一致，PyTorch 为 InterpolationMode 枚举类, Paddle 为 int 或 string，需要转写。        |
| max_size       | -                | 允许的最长边的最大值，Paddle 无此参数，暂无转写方式。                             |
| antialias      | -                | 是否应用抗锯齿处理，Paddle 无此参数，暂无转写方式。                             |
| -              | keys             | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |


### 转写示例
#### interpolation：插值的方法
```python
# PyTorch 写法
transform = torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
resized_img = transform(img)

# Paddle 写法
transform = paddle.vision.transforms.Resize(size=(224, 224), interpolation='bilinear')
resized_img = transform(img)
```
