## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.adjust_contrast

### [torchvision.transforms.functional.adjust_contrast](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.adjust_contrast.html#torchvision.transforms.functional.adjust_contrast)

```python
torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
```

### [paddle.vision.transforms.adjust_contrast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/adjust_contrast_cn.html#paddle/vision/transforms/adjust_contrast_cn#cn-api-paddle-vision-transforms-adjust_contrast)

```python
paddle.vision.transforms.adjust_contrast(img, contrast_factor)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)

# Paddle 写法
result = paddle.vision.transforms.adjust_contrast(img, contrast_factor)
```
