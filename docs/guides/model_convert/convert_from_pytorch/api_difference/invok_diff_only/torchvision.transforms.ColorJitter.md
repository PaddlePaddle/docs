## [ 仅 API 调用方式不一致 ]torchvision.transforms.ColorJitter
### [torchvision.transforms.ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)
```python
torchvision.transforms.ColorJitter(brightness: Union[float, Tuple[float, float]] = 0, contrast: Union[float, Tuple[float, float]] = 0, saturation: Union[float, Tuple[float, float]] = 0, hue: Union[float, Tuple[float, float]] = 0)
```

### [paddle.vision.transforms.ColorJitter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/ColorJitter_cn.html)
```python
paddle.vision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0, keys=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
