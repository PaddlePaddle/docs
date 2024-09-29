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

| torchvision.transforms.ToTensor | paddle.vision.transforms.ToTensor          | 备注                                                         |
|---------------------------------|--------------------------------------------|--------------------------------------------------------------|
| -                               | data_format (str，可选，默认值='CHW')      | Paddle 特有参数，指定返回的 Tensor 格式，可选 `'CHW'` 或 `'HWC'`。 |
| -                               | keys (List[str] \| Tuple[str], 可选)       | Paddle 特有参数。                       |
