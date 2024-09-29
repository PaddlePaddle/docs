## [paddle 参数更多]torchvision.transforms.CenterCrop

### [torchvision.transforms.CenterCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html)

```python
torchvision.transforms.CenterCrop(size: int | list | tuple)
```

### [paddle.vision.transforms.CenterCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/CenterCrop_cn.html)

```python
paddle.vision.transforms.CenterCrop(size: int | list | tuple, keys: list[str] | tuple[str] = None)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | paddle | 备注                                     |
| --------------------------------- | ------------------------------------- | ---------------------------------------- |
| size             | size            | 输出图像的形状大小。 |
| -                                 | keys (optional)    | Paddle 支持 `keys` 参数。 |
