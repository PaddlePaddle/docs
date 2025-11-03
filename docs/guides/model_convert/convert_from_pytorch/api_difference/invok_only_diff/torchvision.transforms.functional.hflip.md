## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.hflip

### [torchvision.transforms.functional.hflip](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.hflip.html#torchvision.transforms.functional.hflip)

```python
torchvision.transforms.functional.hflip(img)
```

### [paddle.vision.transforms.hflip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/hflip_cn.html#paddle/vision/transforms/hflip_cn#cn-api-paddle-vision-transforms-hflip)

```python
paddle.vision.transforms.hflip(img)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.hflip(img)

# Paddle 写法
result = paddle.vision.transforms.hflip(img)

```
