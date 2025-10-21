## [ 仅 API 调用方式不一致 ]torchvision.transforms.Pad

### [torchvision.transforms.Pad](https://pytorch.org/vision/main/generated/torchvision.transforms.Pad.html)

```python
torchvision.transforms.Pad(
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: str = 'constant'
)
```

### [paddle.vision.transforms.Pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Pad__upper_cn.html#pad)
```python
paddle.vision.transforms.Pad(
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Union[int, List[int], Tuple[int, ...]] = 0,
    padding_mode: str = 'constant',
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

两者功能一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：
