## [输入参数类型不一致]torchvision.transforms.RandomResizedCrop

### [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)

```python
torchvision.transforms.RandomResizedCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = True
)
```

### [paddle.vision.transforms.RandomResizedCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomResizedCrop_cn.html)

```python
paddle.vision.transforms.RandomResizedCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    scale: Union[List[float], Tuple[float, float]] = (0.08, 1.0),
    ratio: Union[List[float], Tuple[float, float]] = (0.75, 1.33),
    interpolation: Union[int, str] = 'bilinear',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但参数类型不一致。

### 参数映射

| torchvision | PaddlePaddle | 备注                             |
| --------------------- | ----------------- | --------------------- |
| size                  | size              | 裁剪后的图片大小。                                           |
| scale                 | scale             | 随机裁剪后图像大小的范围。                |
| ratio                 | ratio             | 裁剪后的目标图像宽高比范围。                                 |
| interpolation         | interpolation     | 插值的方法，两者类型不一致，PyTorch 为 InterpolationMode 枚举类, Paddle 为 int 或 string，需要转写。 |
| antialias             | -                 | 是否应用抗锯齿效果，Paddle 无此参数，暂无转写方式。       |
| -                     | keys              | 输入的类型，PyTorch 无此参数，Paddle 保持默认即可。     |

### 转写示例
#### interpolation：插值的方法
```python
# PyTorch 写法
transform = torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
cropped_img = transform(img)

# Paddle 写法
transform = paddle.vision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation='bilinear')
cropped_img = transform(img)
```
