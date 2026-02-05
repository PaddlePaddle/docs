## [ 仅 API 调用方式不一致 ]torchvision.transforms.RandomCrop

### [torchvision.transforms.RandomCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop)

```python
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
```

### [paddle.vision.transforms.RandomCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/RandomCrop_cn.html#paddle.vision.transforms.RandomCrop)

```python
paddle.vision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant', keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
crop = torchvision.transforms.RandomCrop(size=size)

# Paddle 写法
crop = paddle.vision.transforms.RandomCrop(size=size)
```
