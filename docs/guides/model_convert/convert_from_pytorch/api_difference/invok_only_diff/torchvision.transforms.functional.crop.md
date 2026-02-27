## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.crop

### [torchvision.transforms.functional.crop](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop)

```python
torchvision.transforms.functional.crop(img, top, left, height, width)
```

### [paddle.vision.transforms.crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/crop_cn.html#paddle.vision.transforms.crop)

```python
paddle.vision.transforms.crop(img, top, left, height, width)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.crop(img, top, left, height, width)

# Paddle 写法
result = paddle.vision.transforms.crop(img, top, left, height, width)
```
