## [ 输入参数类型不一致 ]torchvision.models.vgg19_bn
### [torchvision.models.vgg19\_bn](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19_bn.html#torchvision.models.vgg19_bn)
```python
torchvision.models.vgg19_bn(pretrained: bool = False, progress: bool = True, *, weights: Optional[VGG19_BN_Weights] = None, **kwargs: Any)
```

### [paddle.vision.models.vgg19](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/vgg19_cn.html#paddle.vision.models.vgg19)
```python
paddle.vision.models.vgg19(pretrained=False, batch_norm=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 VGG19_BN_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| pretrained  | pretrained            | 是否加载预训练权重。torchvision 在 0.13+ 弃用此参数。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| -           | batch_norm   | 是否使用批归一化，PyTorch 无此参数，Paddle 应设置为 True。 |
| **kwargs      | **kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.vgg19_bn(pretrained=True, batch_norm=True)
```
