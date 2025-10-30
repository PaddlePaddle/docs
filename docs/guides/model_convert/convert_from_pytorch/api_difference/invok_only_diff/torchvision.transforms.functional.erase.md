## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.erase

### [torchvision.transforms.functional.erase](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.erase.html#torchvision.transforms.functional.erase)

```python
torchvision.transforms.functional.erase(img, i, j, h, w, v, inplace)
```

### [paddle.vision.transforms.erase](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/erase_cn.html#paddle/vision/transforms/erase_cn#cn-api-paddle-vision-transforms-erase)

```python
paddle.vision.transforms.erase(img, i, j, h, w, v, inplace)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.erase(img, i, j, h, w, v, inplace)

# Paddle 写法
result = paddle.vision.transforms.erase(img, i, j, h, w, v, inplace)
```
