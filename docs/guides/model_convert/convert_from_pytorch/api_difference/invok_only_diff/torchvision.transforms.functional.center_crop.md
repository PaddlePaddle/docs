## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.center_crop

### [torchvision.transforms.functional.center_crop](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.center_crop.html#torchvision.transforms.functional.center_crop)

```python
torchvision.transforms.functional.center_crop(img, output_size)
```

### [paddle.vision.transforms.center_crop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/center_crop_cn.html#paddle/vision/transforms/center_crop_cn#cn-api-paddle-vision-transforms-center_crop)

```python
paddle.vision.transforms.center_crop(img, output_size)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.center_crop(img, output_size)

# Paddle 写法
result = paddle.vision.transforms.center_crop(img, output_size)

```
