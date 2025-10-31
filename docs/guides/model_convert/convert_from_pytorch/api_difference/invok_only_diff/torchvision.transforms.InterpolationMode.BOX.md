## [ 仅 API 调用方式不一致 ]torchvision.transforms.InterpolationMode.BOX

### [torchvision.transforms.InterpolationMode.BOX](https://pytorch.org/vision/stable/generated/torchvision.transforms.InterpolationMode.html#torchvision.transforms.InterpolationMode.BOX)

```python
torchvision.transforms.InterpolationMode.BOX
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
## PyTorch 写法
mode = torchvision.transforms.InterpolationMode.BOX

## Paddle 写法
mode = 'box'
```
