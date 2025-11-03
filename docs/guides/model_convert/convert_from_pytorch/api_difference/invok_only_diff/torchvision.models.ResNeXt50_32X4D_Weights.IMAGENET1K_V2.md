## [ 仅 API 调用方式不一致 ]torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2

### [torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2](https://pytorch.org/vision/stable/models/generated/ResNeXt50_32X4D_Weights.html#torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

```python
torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2

## Paddle 写法
mode = 'IMAGENET1K_V2'
```
