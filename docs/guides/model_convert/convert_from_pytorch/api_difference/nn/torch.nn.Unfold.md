## [ 仅参数名不一致 ]torch.nn.Unfold
### [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=nn+unfold#torch.nn.Unfold)

```python
torch.nn.Unfold(kernel_size,
                dilation=1,
                padding=0,
                stride=1)
```

### [paddle.nn.Unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Unfold_cn.html#unfold)

```python
paddle.nn.Unfold(kernel_sizes=[3, 3],
                    strides=1,
                    paddings=1,
                    dilations=1,
                    name=None)
```

其中功能一致, 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| kernel_size   | kernel_sizes  | 卷积核的尺寸。                    |
| dilation      | dilations     | 卷积膨胀。                        |
| padding       | paddings     | 每个维度的扩展，仅参数名不一致。  |
| stride        | strides      | 卷积步长，仅参数名不一致。        |

### 转写示例
``` python
# PyTorch 写法：
unfold = nn.Unfold(kernel_size=(2, 3))

# Paddle 写法
unfold = nn.Unfold(kernel_sizes=[2, 3])
```
