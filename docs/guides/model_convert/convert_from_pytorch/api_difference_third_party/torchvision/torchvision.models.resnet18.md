## [输入参数类型不一致]torchvision.models.resnet18

### [torchvision.models.resnet18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

```python
torchvision.models.resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.resnet18](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/resnet18_cn.html)

```python
paddle.vision.models.resnet18(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 ResNet18_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
