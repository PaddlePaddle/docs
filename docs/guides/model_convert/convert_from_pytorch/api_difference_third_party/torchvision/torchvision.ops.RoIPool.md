## [参数完全一致]torchvision.ops.RoIPool

### [torchvision.ops.RoIPool](https://pytorch.org/vision/main/generated/torchvision.ops.RoIPool.html)

```python
torchvision.ops.RoIPool(output_size: None, spatial_scale: float)
```

### [paddle.vision.ops.RoIPool](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/RoIPool_cn.html)

```python
paddle.vision.ops.RoIPool(output_size, spatial_scale=1.0)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| output_size                           | output_size        | 池化后输出的尺寸。|
| spatial_scale                         | spatial_scale      | 空间比例因子。|
