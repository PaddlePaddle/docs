## [paddle 参数更多]torchvision.transforms.RandomCrop

### [torchvision.transforms.RandomCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html)

```python
torchvision.transforms.RandomCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    padding: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    pad_if_needed: bool = False,
    fill: Union[float, Tuple[float, ...]] = 0,
    padding_mode: str = 'constant'
)
```

### [paddle.vision.transforms.RandomCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomCrop_cn.html)

```python
paddle.vision.transforms.RandomCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    padding: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    pad_if_needed: bool = False,
    fill: Union[float, Tuple[float, ...]] = 0,
    padding_mode: str = 'constant',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注                                                         |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| size              | size           | 裁剪后的图片大小。                                           |
| padding  | padding  |  对图像四周外边进行填充。                           |
| pad_if_needed                | pad_if_needed                  | 是否在裁剪前进行填充以满足大小要求。                         |
| fill         | fill                 | 用于填充的像素值。   |
| padding_mode                  | padding_mode                    | 填充模式，支持 'constant', 'edge', 'reflect', 'symmetric'。 |
| -                                  | keys  | Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。            |
