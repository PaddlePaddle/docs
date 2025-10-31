## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.BILINEAR

### [torchvision.transforms.InterpolationMode.BILINEAR](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.BILINEAR)

```python
torchvision.transforms.InterpolationMode.BILINEAR
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.BILINEAR

## Paddle 写法
mode = 'bilinear'
```
