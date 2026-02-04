## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.to_grayscale

### [torchvision.transforms.functional.to_grayscale](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.to_grayscale.html#torchvision.transforms.functional.to_grayscale)

```python
torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
```

### [paddle.vision.transforms.to\_grayscale](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/to_grayscale_cn.html#paddle.vision.transforms.to_grayscale)

```python
paddle.vision.transforms.to_grayscale(img, num_output_channels=1)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

# Paddle 写法
result = paddle.vision.transforms.to_grayscale(img, num_output_channels=1)
```
