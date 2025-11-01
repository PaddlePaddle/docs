## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.LANCZOS

### [torchvision.transforms.InterpolationMode.LANCZOS](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.LANCZOS)

```python
torchvision.transforms.InterpolationMode.LANCZOS
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.LANCZOS

## Paddle 写法
mode = 'lanczos'
```
