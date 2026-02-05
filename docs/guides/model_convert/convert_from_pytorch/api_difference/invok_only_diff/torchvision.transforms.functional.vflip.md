## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.vflip

### [torchvision.transforms.functional.vflip](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.vflip.html#torchvision.transforms.functional.vflip)

```python
torchvision.transforms.functional.vflip(img)
```

### [paddle.vision.transforms.vflip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/vflip_cn.html#paddle.vision.transforms.vflip)

```python
paddle.vision.transforms.vflip(img)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.vflip(img)

# Paddle 写法
result = paddle.vision.transforms.vflip(img)

```
