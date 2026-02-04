## [ 输入参数类型不一致 ]torchvision.models.inception_v3
### [torchvision.models.inception\_v3](https://pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html#torchvision.models.inception_v3)
```python
torchvision.models.inception_v3(pretrained: bool = False, progress: bool = True, *, weights: Optional[Inception_V3_Weights] = None, **kwargs: Any)
```

### [paddle.vision.models.inception\_v3](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/models/inception_v3_cn.html#paddle.vision.models.inception_v3)
```python
paddle.vision.models.inception_v3(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| pretrained     | pretrained   | 是否使用预训练权重，PyTorch 已弃用此参数。 |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 Inception_V3_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| **kwargs      | **kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.inception_v3(pretrained=True)
```
