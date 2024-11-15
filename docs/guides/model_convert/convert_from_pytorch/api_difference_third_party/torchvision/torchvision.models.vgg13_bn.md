## [输入参数类型不一致]torchvision.models.vgg13_bn

### [torchvision.models.vgg13_bn](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg13_bn.html)

```python
torchvision.models.vgg13_bn(*, weights: Optional[VGG13_BN_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.vgg13](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/vgg13_cn.html)

```python
paddle.vision.models.vgg13(pretrained=False, batch_norm=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 VGG13_BN_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| -           | batch_norm   | 是否使用批归一化，PyTorch 无此参数，Paddle 应设置为 True。 |
| kwargs      | kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.vgg13_bn(weights=torchvision.models.VGG13_BN_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.vgg13_bn(pretrained=True, batch_norm=True)
```
