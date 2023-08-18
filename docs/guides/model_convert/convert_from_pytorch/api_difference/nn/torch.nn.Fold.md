## [ 仅参数名不一致 ]torch.nn.Fold
### [torch.nn.Fold](https://pytorch.org/docs/stable/generated/torch.nn.Fold.html?highlight=nn+fold#torch.nn.Fold)

```python
torch.nn.Fold(output_size,
                kernel_size,
                dilation=1,
                padding=0,
                stride=1)
```

### [paddle.nn.Fold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Fold_cn.html#fold)

```python
paddle.nn.Fold(output_sizes,
                kernel_sizes,
                dilations=1,
                paddings=0,
                strides=1,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| output_size   | output_sizes | 输出尺寸，整数或者整型列表。                   |
| kernel_size   | kernel_sizes | 卷积核大小，整数或者整型列表。                  |
| dilation      | dilations    | 卷积膨胀，整型列表或者整数。                   |
| padding       | paddings     | 每个维度的扩展，整数或者整型列表。              |
| stride        | strides      | 步长大小，整数或者整型列表。                   |
