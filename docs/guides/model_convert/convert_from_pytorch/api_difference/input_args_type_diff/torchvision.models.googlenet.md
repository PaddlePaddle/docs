## [ 输入参数类型不一致 ]torchvision.models.googlenet
### [torchvision.models.googlenet](https://pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html)
```python
torchvision.models.googlenet(pretrained: bool = False, progress: bool = True, *, weights: Optional[GoogLeNet_Weights] = None, **kwargs: Any)
```

### [paddle.vision.models.googlenet](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/googlenet_cn.html)
```python
paddle.vision.models.googlenet(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 GoogLeNet_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| pretrained  | pretrained            | 是否加载预训练权重。torchvision 在 0.13+ 弃用此参数。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| **kwargs      | **kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.googlenet(pretrained=True)
```
