## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.pad

### [torchvision.transforms.functional.pad](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.pad.html#torchvision.transforms.functional.pad)

```python
torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
```

### [paddle.vision.transforms.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/pad_cn.html#paddle.vision.transforms.pad)

```python
paddle.vision.transforms.pad(img, padding, fill=0, padding_mode='constant')
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.pad(
    img=img, padding=padding, fill=fill, padding_mode=padding_mode
)

# Paddle 写法
result = paddle.vision.transforms.pad(
    img=img, padding=padding, fill=fill, padding_mode=padding_mode
)

```
