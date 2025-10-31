## [ 仅 API 调用方式不一致 ]torchvision.models.VGG19_BN_Weights.DEFAULT

### [torchvision.models.VGG19_BN_Weights.DEFAULT](https://pytorch.org/vision/stable/models/generated/VGG19_BN_Weights.html#torchvision.models.VGG19_BN_Weights.DEFAULT)

```python
torchvision.models.VGG19_BN_Weights.DEFAULT
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.models.VGG19_BN_Weights.DEFAULT

## Paddle 写法
mode = 'DEFAULT'
```
