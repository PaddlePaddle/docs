## [输入参数类型不一致]torchvision.models.resnet34

### [torchvision.models.resnet34](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)

```python
torchvision.models.resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.resnet34](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/resnet34_cn.html)

```python
paddle.vision.models.resnet34(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 ResNet34_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.resnet34(pretrained=True)
```
