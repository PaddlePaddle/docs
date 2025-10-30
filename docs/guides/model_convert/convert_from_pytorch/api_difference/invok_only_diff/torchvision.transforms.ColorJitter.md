## [ 仅 API 调用方式不一致 ]torchvision.transforms.ColorJitter

### [torchvision.transforms.ColorJitter](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter)

```python
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```

### [paddle.vision.transforms.ColorJitter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/ColorJitter_cn.html#paddle/vision/transforms/ColorJitter_cn#cn-api-paddle-vision-transforms-ColorJitter)

```python
paddle.vision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0, keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
jitter = torchvision.transforms.ColorJitter(
    brightness=0.2, contrast=0.3, saturation=0.4, hue=0
)

# Paddle 写法
jitter = paddle.vision.transforms.ColorJitter(
    brightness=0.2, contrast=0.3, saturation=0.4, hue=0
)
```
