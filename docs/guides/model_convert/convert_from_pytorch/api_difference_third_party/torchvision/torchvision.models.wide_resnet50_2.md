## [输入参数类型不一致]torchvision.models.wide_resnet50_2

### [torchvision.models.wide_resnet50_2](https://pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html)

```python
torchvision.models.wide_resnet50_2(*, weights: Optional[WideResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.wide_resnet50_2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/wide_resnet50_2_cn.html)

```python
paddle.vision.models.wide_resnet50_2(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 WideResNet50_2_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
