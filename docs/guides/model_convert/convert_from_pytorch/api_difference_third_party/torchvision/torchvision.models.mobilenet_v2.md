## [输入参数类型不一致]torchvision.models.mobilenet_v2

### [torchvision.models.mobilenet_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)

```python
torchvision.models.mobilenet_v2(*, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.mobilenet_v2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/mobilenet_v2_cn.html)

```python
paddle.vision.models.mobilenet_v2(pretrained=False, scale=1.0, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 MobileNet_V2_Weights 类型，需要转写。|
| -           | scale        | 模型通道数的缩放比例，PyTorch 无此参数，Paddle 保持默认即可。 |
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
