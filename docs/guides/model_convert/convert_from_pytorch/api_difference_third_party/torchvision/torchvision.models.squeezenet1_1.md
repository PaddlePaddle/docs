## [输入参数类型不一致]torchvision.models.squeezenet1_1

### [torchvision.models.squeezenet1_1](https://pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_1.html)

```python
torchvision.models.squeezenet1_1(*, weights: Optional[SqueezeNet1_1_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.squeezenet1_1](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/squeezenet1_1_cn.html)

```python
paddle.vision.models.squeezenet1_1(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 SqueezeNet1_1_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
