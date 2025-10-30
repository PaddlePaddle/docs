## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.HAMMING

### [torchvision.transforms.InterpolationMode.HAMMING](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.HAMMING)

```python
torchvision.transforms.InterpolationMode.HAMMING
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.HAMMING

## Paddle 写法
mode = 'hamming'
```
