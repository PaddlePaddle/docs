## [ 仅 API 调用方式不一致 ]torchvision.models.DenseNet201_Weights.IMAGENET1K_V1

### [torchvision.models.DenseNet201_Weights.IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html#torchvision.models.DenseNet201_Weights)

```python
torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1

## Paddle 写法
mode = 'IMAGENET1K_V1'
```
