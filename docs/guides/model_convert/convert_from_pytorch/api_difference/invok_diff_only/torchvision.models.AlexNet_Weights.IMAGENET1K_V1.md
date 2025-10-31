## [ 仅 API 调用方式不一致 ]torchvision.models.AlexNet_Weights.IMAGENET1K_V1

### [torchvision.models.AlexNet_Weights.IMAGENET1K_V1](https://pytorch.org/vision/stable/models/generated/AlexNet_Weights.html#torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

```python
torchvision.models.AlexNet_Weights.IMAGENET1K_V1
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.models.AlexNet_Weights.IMAGENET1K_V1

## Paddle 写法
mode = 'IMAGENET1K_V1'
```
