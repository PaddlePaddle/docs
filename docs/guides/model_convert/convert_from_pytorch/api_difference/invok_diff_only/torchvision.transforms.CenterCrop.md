## [ 仅 API 调用方式不一致 ]torchvision.transforms.CenterCrop
### [torchvision.transforms.CenterCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.CenterCrop.html)
```python
torchvision.transforms.CenterCrop(
    size: Union[int, List[int], Tuple[int, ...]]
)
```

### [paddle.vision.transforms.CenterCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/CenterCrop_cn.html)
```python
paddle.vision.transforms.CenterCrop(
    size: Union[int, List[int], Tuple[int, ...]],
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：
