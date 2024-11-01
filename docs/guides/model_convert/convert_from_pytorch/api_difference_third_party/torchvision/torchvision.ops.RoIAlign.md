## [torch 参数更多]torchvision.ops.RoIAlign

### [torchvision.ops.RoIAlign](https://pytorch.org/vision/main/generated/torchvision.ops.RoIAlign.html)

```python
torchvision.ops.RoIAlign(output_size: None, spatial_scale: float, sampling_ratio: int, aligned: bool = False)
```

### [paddle.vision.ops.RoIAlign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/RoIAlign_cn.html)

```python
paddle.vision.ops.RoIAlign(output_size, spatial_scale=1.0)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| torchvision                           | PaddlePaddle       | 备注      |
| ------------------------------------- | ------------------ | -------- |
| output_size                           | output_size        | 池化后输出的尺寸。|
| spatial_scale                         | spatial_scale      | 空间比例因子。|
| sampling_ratio                        | -                  | 用于计算每个池化输出条柱的输出值的采样点数，Paddle 无此参数，暂无转写方式。|
| aligned                               | -                  | 像素移动框是否将其坐标移动-0.5，Paddle 无此参数，暂无转写方式。|
