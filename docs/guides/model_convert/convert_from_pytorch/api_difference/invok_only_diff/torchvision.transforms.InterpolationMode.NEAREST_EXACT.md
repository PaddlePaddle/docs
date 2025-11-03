## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.NEAREST_EXACT

### [torchvision.transforms.InterpolationMode.NEAREST_EXACT](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.NEAREST_EXACT)

```python
torchvision.transforms.InterpolationMode.NEAREST_EXACT
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.NEAREST_EXACT

## Paddle 写法
mode = 'nearest_exact'
```
