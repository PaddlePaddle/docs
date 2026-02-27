## [ 仅 API 调用方式不一致 ]torchvision.transforms.Grayscale

### [torchvision.transforms.Grayscale](https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale)

```python
torchvision.transforms.Grayscale(num_output_channels)
```

### [paddle.vision.transforms.Grayscale](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Grayscale_cn.html#paddle.vision.transforms.Grayscale)

```python
paddle.vision.transforms.Grayscale(num_output_channels, keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
img = torch.tensor(
    [[[0.2, 0.4], [0.6, 0.8]], [[0.1, 0.3], [0.5, 0.7]], [[0.0, 0.2], [0.4, 0.6]]]
)
result = grayscale(img)

# Paddle 写法
grayscale = paddle.vision.transforms.Grayscale(num_output_channels=1)
img = paddle.tensor(
    [[[0.2, 0.4], [0.6, 0.8]], [[0.1, 0.3], [0.5, 0.7]], [[0.0, 0.2], [0.4, 0.6]]]
)
result = grayscale(img)

```
