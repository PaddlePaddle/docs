## [paddle 参数更多]torchvision.ops.deform_conv2d

### [torchvision.ops.deform_conv2d](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html)

```python
torchvision.ops.deform_conv2d(input: Tensor, offset: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), mask: Optional[Tensor] = None)
```

### [paddle.vision.ops.deform_conv2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vision/ops/deform_conv2d_cn.html)

```python
paddle.vision.ops.deform_conv2d(x, offset, weight, bias=None, stride=1, padding=0, dilation=1, deformable_groups=1, groups=1, mask=None, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| torchvision | PaddlePaddle | 备注    |
| --------------------------- | ------------------------------ | --------------------- |
| input                       | x                            | 输入数据，仅参数名不一致。       |
| offset                      | offset                       | 可变形卷积层的输入坐标偏移。       |
| weight                      | weight                       | 卷积核参数。       |
| bias                        | bias                         | 可变形卷积偏置项。       |
| stride                      | stride                       | 步长大小。       |
| padding                     | padding                      | 填充大小。       |
| dilation                    | dilation                     | 空洞大小。       |
| -                           | deformable_groups            | 可变形卷积组数，PyTorch 无此参数，Paddle 保持默认即可。       |
| -                           | groups                       | 二维卷积层的组数，PyTorch 无此参数，Paddle 保持默认即可。       |
| mask                        | mask                         | 可变形卷积层的输入掩码。       |
