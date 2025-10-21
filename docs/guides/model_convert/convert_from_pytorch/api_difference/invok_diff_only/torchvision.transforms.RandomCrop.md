## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomCrop
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
