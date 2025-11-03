## [ 仅 API 调用方式不一致 ]torchvision.transforms.ToTensor

### [torchvision.transforms.ToTensor](https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor)

```python
torchvision.transforms.ToTensor()
```

### [paddle.vision.transforms.ToTensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/ToTensor_cn.html#paddle/vision/transforms/ToTensor_cn#cn-api-paddle-vision-transforms-ToTensor)

```python
paddle.vision.transforms.ToTensor(data_format='CHW', keys=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
transform = torchvision.transforms.ToTensor()

# Paddle 写法
transform = paddle.vision.transforms.ToTensor()
```
