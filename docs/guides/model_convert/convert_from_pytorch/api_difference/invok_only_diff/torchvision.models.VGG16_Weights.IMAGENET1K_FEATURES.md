## [ 仅 API 调用方式不一致 ]torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES

### [torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES](https://pytorch.org/vision/stable/models/generated/VGG16_Weights.html#torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)

```python
torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES

## Paddle 写法
mode = 'IMAGENET1K_FEATURES'
```
