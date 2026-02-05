## [ 输入参数类型不一致 ]torchvision.models.wide_resnet50_2
### [torchvision.models.wide\_resnet50\_2](https://pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html#torchvision.models.wide_resnet50_2)
```python
torchvision.models.wide_resnet50_2(pretrained: bool = False, progress: bool = True, *, weights: Optional[WideResNet50_2_Weights] = None, **kwargs: Any)
```

### [paddle.vision.models.wide\_resnet50\_2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/wide_resnet50_2_cn.html#paddle.vision.models.wide_resnet50_2)
```python
paddle.vision.models.wide_resnet50_2(pretrained=False, **kwargs)
```


两者功能一致，但参数类型不一致。 具体而言，PyTorch 框架中内置了两种预训练权重模型，分别为 Wide_ResNet50_2_Weights.IMAGENET1K_V1 和 Wide_ResNet50_2_Weights.IMAGENET1K_V2。而 PaddlePaddle 框架中仅内置了一种预训练权重模型。
在使用模型转换工具 PaConvert 时，无论用户在 PyTorch 中选择使用哪种预训练权重类型，均会统一转换为 PaddlePaddle 中的 pretrained=True 参数配置。

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 WideResNet50_2_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| pretrained  | pretrained            | 是否加载预训练权重。torchvision 在 0.13+ 弃用此参数。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| **kwargs      | **kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.wide_resnet50_2(weights=torchvision.models.WideResNet50_2_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.wide_resnet50_2(pretrained=True)
```
