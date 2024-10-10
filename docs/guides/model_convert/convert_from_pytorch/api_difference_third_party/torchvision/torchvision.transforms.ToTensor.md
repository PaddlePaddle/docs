## [paddle 参数更多]torchvision.transforms.ToTensor

### [torchvision.transforms.ToTensor](https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor)

```python
torchvision.transforms.ToTensor()
```

### [paddle.vision.transforms.ToTensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/transforms/ToTensor_cn.html#totensor)

```python
paddle.vision.transforms.ToTensor(data_format: str = 'CHW', keys: List[str] | Tuple[str] = None)
```

两者功能基本一致，但 Paddle 相比 torchvision 支持更多参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle          | 备注                                                         |
|---------------------------------|--------------------------------------------|--------------------------------------------------------------|
| -                               | data_format       |  PyTorch 无此参数，Paddle 保持默认即可。 |
| -                               | keys        |        Paddle 支持 `keys` 参数，PyTorch 无此参数，Paddle 保持默认即可。           |
