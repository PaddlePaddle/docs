## [输入参数类型不一致]torchvision.models.wide_resnet101_2

### [torchvision.models.wide_resnet101_2](https://pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html)

```python
torchvision.models.wide_resnet101_2(*, weights: Optional[WideResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.wide_resnet101_2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/wide_resnet101_2_cn.html)

```python
paddle.vision.models.wide_resnet101_2(pretrained=False, **kwargs)
```

两者功能一致，但参数类型不一致。 具体而言，PyTorch 框架中内置了两种预训练权重模型，分别为 Wide_ResNet101_2_Weights.IMAGENET1K_V1 和 Wide_ResNet101_2_Weights.IMAGENET1K_V2。而 PaddlePaddle 框架中仅内置了一种预训练权重模型。
在使用模型转换工具 PaConvert 时，无论用户在 PyTorch 中选择使用哪种预训练权重类型，均会统一转换为 PaddlePaddle 中的 pretrained=True 参数配置。

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 WideResNet101_2_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
