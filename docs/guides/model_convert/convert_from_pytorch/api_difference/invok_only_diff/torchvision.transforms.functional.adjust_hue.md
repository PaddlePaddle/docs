## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.adjust_hue

### [torchvision.transforms.functional.adjust_hue](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_hue.html#torchvision.transforms.functional.adjust_hue)

```python
torchvision.transforms.functional.adjust_hue(img, hue_factor)
```

### [paddle.vision.transforms.adjust\_hue](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_hue_cn.html#paddle.vision.transforms.adjust_hue)

```python
paddle.vision.transforms.adjust_hue(img, hue_factor)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.adjust_hue(
    Image.new("RGB", (2, 2), color=(100, 100, 100)), 0.25
)

# Paddle 写法
result = paddle.vision.transforms.adjust_hue(
    Image.new("RGB", (2, 2), color=(100, 100, 100)), 0.25
)

```
