## [paddle 参数更多]torchvision.transforms.RandomCrop

### [torchvision.transforms.RandomCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html)

```python
torchvision.transforms.RandomCrop(size: int | list | tuple, padding: int | list | tuple = None, pad_if_needed: bool = False, fill: float | tuple = 0, padding_mode: str = 'constant')
```

### [paddle.vision.transforms.RandomCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomCrop_cn.html)

```python
paddle.vision.transforms.RandomCrop(size: int | list | tuple, padding: int | list | tuple = None, pad_if_needed: bool = False, fill: float | tuple = 0, padding_mode: str = 'constant', keys: list[str] | tuple[str] = None)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| size (int or list or tuple)             | size (int or list or tuple)          | 裁剪后的图片大小。                                           |
| padding (int or list or tuple, optional) | padding (int or list or tuple, optional) | 两者均支持单个整数或序列进行填充。                           |
| pad_if_needed (bool)               | pad_if_needed (bool)                 | 是否在裁剪前进行填充以满足大小要求。                         |
| fill (float or tuple)        | fill (float or tuple)                | 用于填充的像素值，仅当 padding_mode 为 'constant' 时有效。   |
| padding_mode (str)                 | padding_mode (str)                   | 填充模式，支持 'constant', 'edge', 'reflect', 'symmetric'。 |
| -                                  | keys (list[str] or tuple[str] = None) | Paddle 支持 `keys` 参数，可用于指定要裁剪的键。            |
