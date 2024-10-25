## [torch 参数更多]torchvision.ops.DeformConv2d

### [torchvision.ops.DeformConv2d](https://pytorch.org/vision/main/generated/torchvision.ops.DeformConv2d.html)

```python
torchvision.ops.DeformConv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True)
```

### [paddle.vision.ops.DeformConv2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/pad_cn.html)

```python
paddle.vision.ops.DeformConv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1, groups=1, weight_attr=None, bias_attr=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注    |
| --------------------------- | ------------------------------ | --------------------- |
| in_channels      | in_channels            | 输入通道数。|
| out_channels     | out_channels        | 由卷积操作产生的输出的通道数。                |
| kernel_size      | kernel_size            | 卷积核大小。|
| stride           | stride               | 步长大小。       |
| padding          | padding              | 填充大小。       |
| dilation         | dilation             | 空洞大小。       |
| -               | deformable_groups    | 可变形卷积组数，PyTorch 无此参数，Paddle 保持默认即可。       |
| -               | groups               | 三维卷积层的组数，PyTorch 无此参数，Paddle 保持默认即可。       |
| bias            | -                    | 可变形卷积偏置项，Paddle 无此参数，暂无转写方式。      |
| -               | weight_attr          | 二维卷积层的可学习参数/权重的属性，PyTorch 无此参数，Paddle 保持默认即可。       |
| -               | bias_attr            | 二维卷积层偏置参数属性对象，PyTorch 无此参数，Paddle 保持默认即可。       |
