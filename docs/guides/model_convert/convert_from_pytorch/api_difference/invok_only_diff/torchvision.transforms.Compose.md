## [ 仅 API 调用方式不一致 ]torchvision.transforms.Compose

### [torchvision.transforms.Compose](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose)

```python
torchvision.transforms.Compose(transforms)
```

### [paddle.vision.transforms.Compose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Compose_cn.html#paddle.vision.transforms.Compose)

```python
paddle.vision.transforms.Compose(transforms)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
composed = torchvision.transforms.Compose(transforms=process)

# Paddle 写法
composed = paddle.vision.transforms.Compose(transforms=process)
```
