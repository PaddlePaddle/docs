## [输入参数类型不一致]torchvision.models.vgg19

### [torchvision.models.vgg19](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html)

```python
torchvision.models.vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.vgg19](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/vgg19_cn.html)

```python
paddle.vision.models.vgg19(pretrained=False, batch_norm=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 VGG19_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| -           | batch_norm   | 是否使用批归一化，PyTorch 无此参数，Paddle 应设置为 False。 |
| kwargs      | kwargs       | 附加的关键字参数。|
