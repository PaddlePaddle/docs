## [ 仅 API 调用方式不一致 ]torchvision.transforms.CenterCrop

### [torchvision.transforms.CenterCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html#torchvision.transforms.CenterCrop)

```python
torchvision.transforms.CenterCrop(size)
```

### [paddle.vision.transforms.CenterCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/CenterCrop_cn.html#paddle.vision.transforms.CenterCrop)

```python
paddle.vision.transforms.CenterCrop(size, keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
center_crop = torchvision.transforms.CenterCrop(size)

# Paddle 写法
center_crop = paddle.vision.transforms.CenterCrop(size)
```
