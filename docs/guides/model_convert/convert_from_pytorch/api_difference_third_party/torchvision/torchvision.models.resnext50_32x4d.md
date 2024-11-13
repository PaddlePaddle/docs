## [输入参数类型不一致]torchvision.models.resnext50_32x4d

### [torchvision.models.resnext50_32x4d](https://pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html)

```python
torchvision.models.resnext50_32x4d(*, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any)
```

### [paddle.vision.models.resnext50_32x4d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/resnext50_32x4d_cn.html)

```python
paddle.vision.models.resnext50_32x4d(pretrained=False, **kwargs)
```

两者功能一致但参数类型不一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注 |
| ----------- | ------------ | ---- |
| weights     | pretrained   | 预训练权重，Paddle 参数 pretrained 为 bool 类型，PyTorch 参数 weights 为 ResNeXt50_32X4D_Weights 类型，需要转写。|
| progress    | -            | 是否显示下载进度条，Paddle 无此参数，暂无转写方式。|
| kwargs      | kwargs       | 附加的关键字参数。|
