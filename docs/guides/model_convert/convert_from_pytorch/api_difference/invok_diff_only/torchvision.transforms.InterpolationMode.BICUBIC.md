## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.BICUBIC

### [torchvision.transforms.InterpolationMode.BICUBIC](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.BICUBIC)

```python
torchvision.transforms.InterpolationMode.BICUBIC
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.BICUBIC

## Paddle 写法
mode = 'bicubic'
```
