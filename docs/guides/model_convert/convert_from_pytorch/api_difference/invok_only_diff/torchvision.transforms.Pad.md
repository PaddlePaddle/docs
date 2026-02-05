## [ 仅 API 调用方式不一致 ]torchvision.transforms.Pad

### [torchvision.transforms.Pad](https://pytorch.org/vision/stable/generated/torchvision.transforms.Pad.html#torchvision.transforms.Pad)

```python
torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
```

### [paddle.vision.transforms.Pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/Pad__upper_cn.html#paddle.vision.transforms.Pad)

```python
paddle.vision.transforms.Pad(padding, fill=0, padding_mode='constant', keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
pad = torchvision.transforms.Pad(padding=2, fill=0, padding_mode="constant")

# Paddle 写法
pad = paddle.vision.transforms.Pad(padding=2, fill=0, padding_mode="constant")
```
