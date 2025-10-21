## [ 仅 API 调用方式不一致 ]torchvision.transforms.Grayscale

### [torchvision.transforms.Grayscale](https://pytorch.org/vision/main/generated/torchvision.transforms.Grayscale.html)
```python
torchvision.transforms.Grayscale(num_output_channels: int = 1)
```

### [paddle.vision.transforms.Grayscale](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Grayscale_cn.html)
```python
paddle.vision.transforms.Grayscale(
    num_output_channels: int = 1,
    keys: Optional[Union[List[str], Tuple[str, ...]]] = None
)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
