## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.adjust_brightness

### [torchvision.transforms.functional.adjust_brightness](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_brightness.html#torchvision.transforms.functional.adjust_brightness)

```python
torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
```

### [paddle.vision.transforms.adjust\_brightness](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_brightness_cn.html#paddle.vision.transforms.adjust_brightness)

```python
paddle.vision.transforms.adjust_brightness(img, brightness_factor)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.adjust_brightness(img, factor)

# Paddle 写法
result = paddle.vision.transforms.adjust_brightness(img, factor)
```
