## [ 参数不一致 ]torch.nn.Unfold
### [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=nn+unfold#torch.nn.Unfold)

```python
torch.nn.Unfold(kernel_size,
                dilation=1,
                padding=0,
                stride=1)
```

### [paddle.nn.Unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Unfold_cn.html#unfold)

```python
paddle.nn.Unfold(kernel_size=[3, 3],
                    strides=1,
                    paddings=1,
                    dilation=1,
                    name=None)
```
其中 Paddle 与 Pytorch 前四个参数所支持的参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size   | kernel_sizes | 卷积核大小， PyTorch 参数类型为 int、tuple(int) 或者 list(int)， Paddle 参数类型为 int 或者 list(int)。   |
| dilation      | dilations    | 卷积膨胀，PyTorch 参数类型为 int、tuple(int) 或者 list(int)， Paddle 参数类型为 int 或者 list(int)。 |
| padding       | paddings     | 每个维度的扩展，PyTorch 参数类型为 int、tuple(int) 或者 list(int)， Paddle 参数类型为 int 或者 list(int)。 |
| stride        | strides      | 步长大小，PyTorch 参数类型为 int、tuple(int) 或者 list(int)， Paddle 参数类型为 int 或者 list(int)。|

### 转写示例
``` python
# PyTorch 写法：
unfold = nn.Unfold(kernel_size=(2, 3))

# Paddle 写法
unfold = nn.Unfold(kernel_size=[2, 3])
```
