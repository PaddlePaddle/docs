## [ 仅 API 调用方式不一致 ]torchvision.transforms.functional.to_tensor

### [torchvision.transforms.functional.to_tensor](https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.to_tensor.html#torchvision.transforms.functional.to_tensor)

```python
torchvision.transforms.functional.to_tensor(pic)
```

### [paddle.vision.transforms.to\_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/to_tensor_cn.html#paddle.vision.transforms.to_tensor)

```python
paddle.vision.transforms.to_tensor(pic, data_format='CHW')
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torchvision.transforms.functional.to_tensor(img)

# Paddle 写法
result = paddle.vision.transforms.to_tensor(img)

```
