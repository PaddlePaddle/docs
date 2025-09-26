## [输入参数类型不一致]torchvision.models.densenet121

### [torchvision.models.densenet121](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)

```python
torchvision.models.densenet121(*, weights: Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.densenet121](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/densenet121_cn.html)

```python
paddle.vision.models.densenet121(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，PyTorch 参数 weights 为 DenseNet121_Weights 枚举类或 String 类型，Paddle 参数 pretrained 为 bool 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| kwargs      | kwargs       | 附加的关键字参数。|

### 转写示例
#### weights: 预训练权重
```python
# PyTorch 写法
torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)

# Paddle 写法
paddle.vision.models.densenet121(pretrained=True)
```
